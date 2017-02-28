#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import math
import os
import re
import time

import numpy as np
import tensorflow as tf

from rorschach.prediction.common.tools import DataDumper
from rorschach.prediction.tensorflow.callbacks import CallbackRunner
from rorschach.prediction.tensorflow.tools import LogPrettifier, TimeParse
from rorschach.utilities import Config, LoggerWrapper


class Seq2Seq(object):

    MODEL_CKPT_PATTERN = re.compile('^model\.ckpt\-[0-9]\.(?:index|meta|data\-[0-9]*\-of\-[0-9]*)')

    def __init__(
        self,
        xseq_len,
        yseq_len,
        xvocab_size,
        yvocab_size,
        emb_dim,
        num_layers
    ):

        self.log = LogPrettifier(LoggerWrapper.load(__name__, LoggerWrapper.SIMPLE))
        self.session = None
        self.callback = None

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

        # Parts of the model
        self.encoder_input_placeholders = None
        self.decoder_input_placeholders = None
        self.label_placeholders = None
        self.keep_probability = None

        self.data_loss = {
            'train': [],
            'validate': []
        }

        self.data_accuracy = []

        self.data_stores = []

        self.epoch_time_start = None

    def build_graph(self):
        # Reset the default graph of Tensorflow here
        tf.reset_default_graph()

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

        # LSTM
        self.keep_probability = tf.placeholder(tf.float32)

        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.BasicLSTMCell(
                self.emb_dim,
                state_is_tuple=True
            ),
            output_keep_prob=self.keep_probability
        )

        # Stacked LSTMs (defined by the number of layers in the model)
        stacked_lstms = tf.nn.rnn_cell.MultiRNNCell(
            [rnn_cell] * self.num_layers,
            state_is_tuple=True
        )

        # Sharing of parameters between training and testing models
        with tf.variable_scope('decoder') as scope:
            self.decode_outputs, self.decode_states = tf.nn.seq2seq.embedding_attention_seq2seq(
                self.encoder_input_placeholders,
                self.decoder_input_placeholders,
                stacked_lstms,
                self.xvocab_size,
                self.yvocab_size,
                self.emb_dim
            )

            scope.reuse_variables()

            # Testing model. Here the output from the previous timestep is fed as input to the next timestep
            self.decode_outputs_test, self.decode_states_test = tf.nn.seq2seq.embedding_attention_seq2seq(
                self.encoder_input_placeholders,
                self.decoder_input_placeholders,
                stacked_lstms,
                self.xvocab_size,
                self.yvocab_size,
                self.emb_dim,
                feed_previous=True
            )

        # Loss for the weights
        loss_weights = [tf.ones_like(
            label,
            dtype=tf.float32
        ) for label in self.label_placeholders]

        # The loss
        self.loss = tf.nn.seq2seq.sequence_loss(
            self.decode_outputs,
            self.label_placeholders,
            loss_weights,
            self.yvocab_size
        )

        # The train op
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=Config.get('predicting.learning-rate')
        ).minimize(self.loss)

    def get_feed(self, X, Y, keep_prob):
        feed_dict = {
            self.encoder_input_placeholders[t]: X[t] for t in range(self.xseq_len)
        }

        feed_dict.update({self.label_placeholders[t]: Y[t] for t in range(self.yseq_len)})
        feed_dict[self.keep_probability] = keep_prob

        return feed_dict

    def train_batch(self):
        batchX, batchY = self.training_set.__next__()

        feed_dict = self.get_feed(
            batchX,
            batchY,
            keep_prob=0.5
        )

        _, train_loss = self.session.run([self.train_op, self.loss], feed_dict)

        return train_loss

    def train(self):
        train_losses = []

        # Divide number of words in the training set on batch size
        num_batches = int(
            math.ceil(Config.get('preprocessing.training-set.size') / Config.get('predicting.batch-size')))

        for i in range(num_batches):
            train_loss = self.train_batch()

            train_losses.append(train_loss)

        # Get the mean loss value
        train_mean_loss = np.mean(train_losses)

        # Add to our store
        self.data_loss['train'].append(float(train_mean_loss))

        self.log.write('- Loss: {:.5f}'.format(train_mean_loss))

    def validate_batch(self):
        batchX, batchY = self.validation_set.__next__()

        feed_dict = self.get_feed(
            batchX,
            batchY,
            keep_prob=1.
        )

        validation_loss, validation_output = self.session.run(
            [
                self.loss,
                self.decode_outputs_test
            ],
            feed_dict
        )

        # Transpose the validation output
        validation_output = np.array(validation_output).transpose([1, 0, 2])

        # Transpose the validate labels
        validate_labels = np.array(batchY).transpose([1, 0])

        # We use argmax to get the output with the highest probability for each character in the output
        validation_predictions = np.argmax(validation_output, axis=2)

        # Create a boolen matrix with true where we predited the correct label and false where we did not
        validation_correct_predictions = np.equal(validation_predictions, validate_labels)

        # Sum the number of correct predictions
        validation_correct = np.sum(validation_correct_predictions)

        # Calculate the number of predictions done in this batch
        validation_total = validation_correct_predictions.shape[0] * validation_correct_predictions.shape[1]

        # Calculate the float value of the accuracy
        validation_accuracy = validation_correct / float(validation_total)

        # Return: loss, calculate accuracy, number of correct predictions and total number of predictions done
        return validation_loss, validation_accuracy, validation_correct, validation_total

    def validate(self):
        validation_losses = []
        validation_accuracies = []

        validation_total_correct = 0
        validation_total_predictions = 0

        # Divide number of words in the validate set on batch size
        num_batches = int(math.ceil(
            Config.get('preprocessing.validate-set.size') / Config.get('predicting.batch-size')))

        for i in range(num_batches):
            validation_loss, validation_accuracy, validation_correct, validation_total = self.validate_batch()

            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_accuracy)

            validation_total_correct += validation_correct
            validation_total_predictions += validation_total

        validation_mean_loss = np.mean(validation_losses)
        validation_mean_accuracy = np.mean(validation_accuracies)

        self.data_loss['validate'].append(float(validation_mean_loss))
        self.data_accuracy.append(float(validation_mean_accuracy))

        if self.callback is not None:
            # Plot the loss
            self.callback.run({
                'loss_train': self.data_loss['train'],
                'loss_validate': self.data_loss['validate'],
                'stores': self.data_stores
            }, CallbackRunner.LOSS)

            # Plot the accuracy
            self.callback.run({
                'accuracy': self.data_accuracy
            }, CallbackRunner.ACCURACY)

        # Output debug
        self.log.write('- Validate loss: {:.5f}'.format(validation_mean_loss))
        self.log.write('- Validate accuracy: {:.5f}'.format(validation_mean_accuracy))
        self.log.write('- Validate correct attributes: {:,} / {:,}'.format(validation_total_correct,
                                                                           validation_total_predictions))

    def test(self):
        pass

    def dump_data(self):
        DataDumper.dump({
            'validate_accuracy': self.data_accuracy,
            'validate_loss': self.data_loss['validate'],
            'train_loss': self.data_loss['train'],
            'stores': self.data_stores
        })

    def run(self):
        saver = tf.train.Saver()

        # Start a session
        if not self.session:
            self.session = tf.Session()

            self.session.run(tf.global_variables_initializer())

        # Loop the epochs
        for epoch in range(Config.get('predicting.epochs')):
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

            # Dump the data
            self.dump_data()

            # Check if we should save our model
            if self.should_save_session():
                self.log.write('- Saving model')

                self.save_session(saver, epoch)

            self.log.write('- Execution: {}'.format(TimeParse.parse(self.epoch_time_start)))

        self.log.write('', LogPrettifier.END)

        # Run final validation
        self.test()

    def should_save_session(self):
        validation_loss = copy.copy(self.data_loss['validate'])
        if len(validation_loss) <= Config.get('predicting.best-results'):
            return False

        # The current (last) loss
        current_loss = validation_loss[-1]

        # The other loss values (without the last)
        validation_loss.pop(-1)

        # If the last value is higher or equal to the highest value in the other losses, we store the weights
        return current_loss <= min(validation_loss)

    def save_session(self, saver, epoch):
        # Add save to plot
        self.data_stores.append(epoch)

        # Delete earlier model files
        dir_content = os.listdir(os.path.join(Config.get('path.output'), Config.get('uid')))
        for content in dir_content:
            if not Seq2Seq.MODEL_CKPT_PATTERN.match(content):
                continue

            os.remove(Config.get_path('path.output', content, fragment=Config.get('uid')))

        # Save the ckpt dump
        saver.save(
            self.session,
            Config.get_path('path.output', 'model.ckpt', fragment=Config.get('uid')),
            global_step=epoch
        )

    def restore_last_session(self):
        # TODO

        '''

        saver = tf.train.Saver()

        sess = tf.Session()

        ckpt = tf.train.get_checkpoint_state()

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        return sess
        '''

        return
