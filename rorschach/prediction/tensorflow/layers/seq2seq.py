#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from rorschach.utilities import Config, LoggerWrapper


class Seq2Seq(object):
    def __init__(
        self,
        xseq_len,
        yseq_len,
        xvocab_size,
        yvocab_size,
        emb_dim,
        num_layers
    ):

        self.log = LoggerWrapper.load(__name__)
        self.session = None

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

        _, loss_v = self.session.run([self.train_op, self.loss], feed_dict)

        self.log.info('Loss %s', loss_v)

        return loss_v

    def test_batch(self):
        batchX, batchY = self.test_set.__next__()

        feed_dict = self.get_feed(
            batchX,
            batchY,
            keep_prob=1.
        )

        loss_v, dec_op_v = self.session.run(
            [
                self.loss,
                self.decode_outputs_test
            ],
            feed_dict
        )

        dec_op_v = np.array(dec_op_v).transpose([1, 0, 2])
        correct = np.array(batchY).transpose([1, 0])

        pred_val = np.argmax(dec_op_v, axis=2)
        correct_pred = np.equal(pred_val, correct)

        accuracy = np.sum(correct_pred) / float(correct_pred.shape[0] * correct_pred.shape[1])

        return loss_v, accuracy

    # TODO fix fixed number of batches(?)
    def test(self, num_batches=16):
        losses = []
        accuracyies = []

        for i in range(num_batches):
            loss_v, accuracy = self.test_batch()
            losses.append(loss_v)
            accuracyies.append(accuracy)

        return np.mean(losses), np.mean(accuracyies)

    def validate(self):
        pass

    def train(self):
        # TODO
        # saver = tf.train.Saver()

        # Start a session
        if not self.session:
            self.session = tf.Session()

            self.session.run(tf.global_variables_initializer())

        # Loop the epochs
        for epoch in range(Config.get('predicting.epochs')):
            self.log.info('Epoch %s', epoch + 1)

            self.train_batch()

            if epoch > 0 and (epoch + 1) % Config.get('predicting.test-interval') == 0:
                # TODO
                # save model to disk
                # saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)

                # Test
                test_loss, test_accuracy = self.test()

                # Output debug
                self.log.info('Test loss: %s', test_loss)
                self.log.info('Test accuracy: %s', test_accuracy)
                self.log.info('=============================================')

        # Run final validation
        self.validate()

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
