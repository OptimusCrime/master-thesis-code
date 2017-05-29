# -*- coding: utf-8 -*-

import glob
import json
import math
import os
import re
import time
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from rorschach._scripts.attention_plot import AttentionPlot
from rorschach.prediction.common import CallbackRunner
from rorschach.prediction.common.callbacks import (DataCallback, EpochIndicatorCallback, PlotterCallback,
                                                   TensorflowSaverCallback)
from rorschach.prediction.tensorflow.tools import LogPrettifier, TimeParse, batch_gen
from rorschach.utilities import Config, JsonConfigEncoder, LoggerWrapper, pickle_data, unpickle_data


class AbstractSeq2seq(ABC):

    MODEl_CKPT_ID_PATTERN = re.compile('^model\.ckpt-(?P<id>[0-9]{1,})\.(?:index|meta|data\-[0-9]*\-of\-[0-9]*)$')

    VALIDATE = 1
    TEST = 2

    def __init__(
        self,
        model,
        xseq_len,
        yseq_len,
        xvocab_size,
        yvocab_size,
        emb_dim,
        num_layers
    ):

        self.log = LogPrettifier(LoggerWrapper.load(__name__, LoggerWrapper.SIMPLE))
        self.model = model
        self.session = None
        self.callback = None
        self.data_container = None

        # Settings
        self.xseq_len = xseq_len
        self.yseq_len = yseq_len

        self.xvocab_size = xvocab_size + 1
        self.yvocab_size = yvocab_size + 1

        self.emb_dim = emb_dim
        self.num_layers = num_layers

        # Sets
        self.training_set = None
        self.test_set = None
        self.validation_set = None

        # Sizes
        self.training_set_size = 0
        self.test_set_size = 0
        self.validation_set_size = 0

        # Parts of the model
        self.encoder_input_placeholders = None
        self.decoder_input_placeholders = None
        self.label_placeholders = None

        self.keep_probability = None
        self.stacked_lstms = None

        self.decode_outputs = None
        self.decode_states = None
        self.decode_outputs_test = None
        self.decode_states_test = None

        self.epoch_time_start = None

        self.loss = None
        self.train_op = None

        # Graphers
        self.graph_writer = None
        self.scalar_writer = None

        # Merge for all tf variables
        self.merged = None
        self.merged_identifier = 0

        self.saver = None

        # Attention mechanism
        self.attention_output = None
        self.attention_output_test = None

        # Context vector
        self.context_vector_output = None
        self.context_vector_output_test = None
        self.special_batch = None

    def register_data_container(self, data_container):
        self.callback = CallbackRunner(data_container)
        self.data_container = data_container

    def build_graph(self):
        # Reset the default graph of Tensorflow here
        tf.reset_default_graph()

        with tf.name_scope('placeholders'):
            # Encoder inputs
            self.encoder_input_placeholders = [tf.placeholder(
                shape=[None, ],
                dtype=tf.int64,
                name='ei_{}'.format(t)
            ) for t in range(self.xseq_len)]

            # Output labels
            self.label_placeholders = [tf.placeholder(
                shape=[None, ],
                dtype=tf.int64,
                name='ei_{}'.format(t)
            ) for t in range(self.yseq_len)]

            # Decoder inputs
            self.decoder_input_placeholders = [tf.zeros_like(
                self.encoder_input_placeholders[0],
                dtype=tf.int64,
                name='GO'
            )] + self.label_placeholders[:-1]

        with tf.name_scope('dropout_probability'):
            # LSTM
            self.keep_probability = tf.placeholder(tf.float32)

        # Build the model itself here
        self.build_model()

        with tf.name_scope('loss'):
            # Loss for the weights
            loss_weights = [tf.ones_like(
                label,
                dtype=tf.float32
            ) for label in self.label_placeholders]

            # The loss
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(
                self.decode_outputs,
                self.label_placeholders,
                loss_weights,
                self.yvocab_size
            )

        with tf.name_scope('optimizer'):
            self.train_op = tf.train.AdamOptimizer(
                learning_rate=Config.get('predicting.learning-rate')
            ).minimize(self.loss)

        if Config.get('various.tensorboard'):
            tf.summary.scalar('keep_probability', self.keep_probability)
            tf.summary.scalar('loss', self.loss)

            self.merged = tf.summary.merge_all()

    @abstractmethod
    def build_model(self):
        pass

    def get_feed(self, x, y, keep_prob):
        feed_dict = {
            self.encoder_input_placeholders[t]: x[t] for t in range(self.xseq_len)
        }

        feed_dict.update({self.label_placeholders[t]: y[t] for t in range(self.yseq_len)})
        feed_dict[self.keep_probability] = keep_prob

        return feed_dict

    def train_batch(self):
        batch_x, batch_y = self.training_set.__next__()

        feed_dict = self.get_feed(
            batch_x,
            batch_y,
            keep_prob=0.8
        )

        if Config.get('various.tensorboard'):
            _, train_loss, summary = self.session.run([self.train_op, self.loss, self.merged], feed_dict)
            self.graph_writer.add_summary(summary, self.merged_identifier)
            self.scalar_writer.add_summary(summary, self.merged_identifier)

            self.merged_identifier += 1
        else:
            _, train_loss = self.session.run([self.train_op, self.loss], feed_dict)

        return train_loss

    def train(self):
        train_losses = []

        # Divide number of words in the training set on batch size
        num_batches = int(
            math.ceil(self.training_set_size / Config.get('predicting.batch-size')))

        for _ in range(num_batches):
            train_loss = self.train_batch()

            train_losses.append(train_loss)

        # Get the mean loss value
        train_mean_loss = np.mean(train_losses)

        # Add to our store
        self.data_container.add_list('train_loss', float(train_mean_loss))

        self.log.write('- Loss: {:.5f}'.format(train_mean_loss))

    def validate_test_batch(self, mode):
        if mode == AbstractSeq2seq.VALIDATE:
            batch_x, batch_y = self.validation_set.__next__()
        else:
            batch_x, batch_y = self.test_set.__next__()

        feed_dict = self.get_feed(
            batch_x,
            batch_y,
            keep_prob=1.
        )

        loss, output = self.session.run(
            [
                self.loss,
                self.decode_outputs_test
            ],
            feed_dict
        )

        # Transpose the validation output
        output = np.array(output).transpose([1, 0, 2])

        # Transpose the validate labels
        labels = np.array(batch_y).transpose([1, 0])

        # We use argmax to get the output with the highest probability for each character in the output
        predictions = np.argmax(output, axis=2)

        # Create a boolen matrix with true where we predited the correct label and false where we did not
        correct_predictions = np.equal(predictions, labels)

        # Sum the number of correct predictions
        correct = np.sum(correct_predictions)

        # Calculate the number of predictions done in this batch
        total = correct_predictions.shape[0] * correct_predictions.shape[1]

        # Calculate the float value of the accuracy
        accuracy = correct / float(total)

        # Return: loss, calculate accuracy, number of correct predictions and total number of predictions done
        return {
            'loss': loss,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }

    def calculate_batch_size(self, mode):
        dividend = self.validation_set_size
        if mode == AbstractSeq2seq.TEST:
            dividend = self.test_set_size

        return int(math.ceil(dividend / Config.get('predicting.batch-size')))

    def validate_test(self, mode):
        losses = []
        accuracies = []

        total_correct = 0
        total_predictions = 0

        for _ in range(self.calculate_batch_size(mode)):
            results = self.validate_test_batch(mode)

            losses.append(results['loss'])
            accuracies.append(results['accuracy'])

            total_correct += results['correct']
            total_predictions += results['total']

        return {
            'losses': losses,
            'accuracies': accuracies,
            'total_correct': total_correct,
            'total_predictions': total_predictions
        }

    def validate(self):
        results = self.validate_test(AbstractSeq2seq.VALIDATE)

        validation_mean_loss = np.mean(results['losses'])
        validation_mean_accuracy = np.mean(results['accuracies'])

        self.data_container.add_list('validate_loss', float(validation_mean_loss))
        self.data_container.add_list('validate_accuracy', float(validation_mean_accuracy))

        if self.callback is not None:
            # Plot the loss
            self.callback.run([PlotterCallback], [PlotterCallback.LOSS])

            # Plot the accuracy
            self.callback.run([PlotterCallback], [PlotterCallback.ACCURACY])

        # Output debug
        self.log.write('- Validate loss: {:.5f}'.format(validation_mean_loss))
        self.log.write('- Validate accuracy: {:.5f}'.format(validation_mean_accuracy))
        self.log.write('- Validate correct attributes: {:,} / {:,}'.format(results['total_correct'],
                                                                           results['total_predictions']))

    def test(self):
        results = self.validate_test(AbstractSeq2seq.TEST)

        data = {
            'loss': np.mean(results['losses']),
            'accuracy': np.mean(results['accuracies'])
        }

        with open(Config.get_path('path.output', 'results.json', fragment=Config.get('uid')), 'w') as outfile:
            json.dump(data, outfile, cls=JsonConfigEncoder)

        print(data)

    def start_train(self):
        # Start a session
        if not self.session:
            self.init_session()

            if Config.get('various.tensorboard'):
                self.graph_writer = tf.summary.FileWriter(
                    Config.get_path('path.output', '', fragment=Config.get('uid')),
                    self.session.graph
                )

            self.scalar_writer = tf.summary.FileWriter(
                Config.get_path('path.output', '', fragment=Config.get('uid'))
            )

        epoch_offset = 0
        if Config.get('general.mode') == 'continue':
            with open(Config.get_path('path.output', 'data.json', fragment=Config.get('uid'))) as json_data:
                data = json.load(json_data)
                if 'epoch' in data:
                    epoch_offset = data['epoch'] + 1

        # Loop the epochs
        for epoch in range(epoch_offset, Config.get('predicting.epochs')):
            self.data_container.set('epoch', epoch)

            self.epoch_time_start = time.time()

            log_type = LogPrettifier.EPOCH
            if epoch == 0:
                log_type = LogPrettifier.FIRST_EPOCH

            # Output epoch information
            self.log.write('Epoch {:d}'.format(epoch + 1), log_type)

            # Train the model
            self.train()

            # Validate the model
            self.validate()

            if self.callback is not None:
                # Dump the data
                self.callback.run([DataCallback])

                # Dump weights
                self.callback.run([TensorflowSaverCallback], None, {
                    'saver': self.saver,
                    'session': self.session,
                    'log': self.log
                })

                # Store epoch indicator
                self.callback.run([EpochIndicatorCallback])

            self.log.write('- Execution: {}'.format(TimeParse.parse(self.epoch_time_start)))

        self.log.write('', LogPrettifier.END)

    def start_test(self):
        self.log.write('Begin test', LogPrettifier.NULL)

        self.init_session(restore=True)
        self.test()

        self.log.write('Finish test', LogPrettifier.NULL)

    def start_predict(self):
        self.log.write('Begin predict', LogPrettifier.NULL)

        self.init_session(restore=True)
        self.predict()

        self.log.write('Finish predict', LogPrettifier.NULL)

    def special_feed(self):
        special_batch = self.create_special_batch()

        batch_x, batch_y = AbstractSeq2seq.special_feed_batch(special_batch)

        return self.get_feed(batch_x, batch_y, keep_prob=1.), batch_x, batch_y

    def create_special_batch(self):
        if Config.get('general.special-dataset') and self.special_batch is not None:
            return self.special_batch

        if Config.get('general.special-dataset'):
            self.special_batch = batch_gen(
                self.model.test_images_transformed,
                self.model.test_labels_transformed,
                1
            )

            if Config.get('general.special-offset') is not None:
                for _ in range(Config.get('general.special-offset')):
                    batch_x, batch_y = self.special_batch.__next__()

            return self.special_batch

        return batch_gen(
            self.model.test_images_transformed,
            self.model.test_labels_transformed,
            1
        )

    @staticmethod
    def special_feed_batch(special_batch):
        if Config.get('general.special-dataset'):
            return special_batch.__next__()

        offset = 1
        batch_x, batch_y = (None, None)
        if Config.get('general.special-offset') is not None:
            offset = Config.get('general.special-offset')

        for _ in range(offset):
            batch_x, batch_y = special_batch.__next__()

        return batch_x, batch_y

    def attention(self):
        if self.attention_output is None or self.attention_output_test is None:
            raise Exception('Special enc/dec not found')

        feed_dict, batch_x, batch_y = self.special_feed()

        attention_values = self.session.run(
            [
                self.attention_output_test
            ],
            feed_dict
        )

        attention_plot = AttentionPlot()
        attention_plot.plot(attention_values[0], batch_x, batch_y)

    def context(self):
        if self.context_vector_output is None or self.context_vector_output_test is None:
            raise Exception('Special context vector not found')

        length = 1
        if Config.get('general.special-length') is not None:
            length = int(Config.get('general.special-length'))

        context_states = []
        for _ in range(length):
            feed_dict, batch_x, batch_y = self.special_feed()

            context_values, output = self.session.run(
                [
                    self.context_vector_output_test,
                    self.decode_outputs_test
                ],
                feed_dict
            )

            context_state = []
            for i in range(len(context_values)):
                context_state.append(context_values[i].h[0])

            context_states.append({
                'state': context_state,
                'input': batch_x
            })

        context_data_file = Config.get_path('path.output', 'context_data.pickl', fragment=Config.get('uid'))
        if os.path.exists(context_data_file):
            context_data = unpickle_data(context_data_file)
        else:
            context_data = {}

        if Config.get('general.special-identifier') not in context_data:
            context_data[Config.get('general.special-identifier')] = []

        context_data[Config.get('general.special-identifier')].extend(context_states)

        pickle_data(context_data, context_data_file)

    def predict(self):
        if Config.get('general.special') == 'attention':
            return self.attention()

        if Config.get('general.special') == 'context':
            return self.context()

        outputs = []
        correct = []

        for _ in range(self.calculate_batch_size(AbstractSeq2seq.TEST)):
            results = self.predict_batch()

            outputs.extend(results['output'])
            correct.extend(results['correct'])

        outputs_arr = np.array(outputs)
        correct_arr = np.array(correct)

        pickle_data({
            'predictions': outputs_arr.tolist(),
            'correct': correct_arr.tolist()
        }, Config.get_path('path.output', 'predictions.pickl', fragment=Config.get('uid')))

    def predict_batch(self):
        batch_x, batch_y = self.test_set.__next__()

        feed_dict = self.get_feed(
            batch_x,
            batch_y,
            keep_prob=1.
        )

        loss, output = self.session.run(
            [
                self.loss,
                self.decode_outputs_test
            ],
            feed_dict
        )

        return {
            'output': np.array(output).transpose([1, 0, 2]),
            'correct': np.array(batch_y).transpose([1, 0])
        }

    def init_session(self, restore=False):
        if restore or Config.get('general.mode') in ['continue', 'test']:
            return self.restore_last_session()

        return self.create_session()

    def create_session(self):
        self.saver = tf.train.Saver(max_to_keep=1)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def restore_last_session(self):
        self.saver = tf.train.Saver(max_to_keep=1)

        self.session = tf.Session()

        checkpoint = AbstractSeq2seq.locate_checkpoint_file()

        self.log.write('Restoring model checkpoint from ' + checkpoint, LogPrettifier.NULL)

        self.saver.restore(self.session, checkpoint)

    @staticmethod
    def locate_checkpoint_file():
        ids = set()
        base_path = Config.get_path('path.output', Config.get('uid'))
        files = glob.glob(base_path + os.sep + '*')

        for file in files:
            filename = file.split(os.sep)[-1]

            matches = re.findall(AbstractSeq2seq.MODEl_CKPT_ID_PATTERN, filename)

            for match in matches:
                if match is not None and len(match) > 0 and match.isdigit():
                    ids.add(int(match))

        if len(ids) == 0:
            raise Exception('No ckpt file found!')

        sorted_ids = sorted(list(ids))

        return Config.get_path('path.output', 'model.ckpt-' + str(sorted_ids[-1]), fragment=Config.get('uid'))
