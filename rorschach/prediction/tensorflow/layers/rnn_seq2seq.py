# -*- coding: utf-8 -*-

import tensorflow as tf

from rorschach.prediction.tensorflow.lib import embedding_rnn_seq2seq
from rorschach.prediction.tensorflow.layers import AbstractSeq2seq
from rorschach.utilities import Config


class RNNSeq2Seq(AbstractSeq2seq):

    def build_model(self):
        with tf.name_scope('rnn'):
            rnn_cell = tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(
                    self.emb_dim,
                    state_is_tuple=True
                ),
                output_keep_prob=self.keep_probability
            )

        with tf.name_scope('stack'):
            # Stacked LSTMs (defined by the number of layers in the model)
            stacked_lstms = tf.contrib.rnn.MultiRNNCell(
                [rnn_cell] * self.num_layers,
                state_is_tuple=True
            )

        if Config.get('general.special') == 'context':
            return self.build_rnn_model(stacked_lstms)

        # Sharing of parameters between training and testing models
        with tf.variable_scope('encdec') as scope:
            self.decode_outputs, self.decode_states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                self.encoder_input_placeholders,
                self.decoder_input_placeholders,
                stacked_lstms,
                self.xvocab_size,
                self.yvocab_size,
                self.emb_dim
            )

            scope.reuse_variables()

            # Testing model. Here the output from the previous timestep is fed as input to the next timestep
            self.decode_outputs_test, self.decode_states_test = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                self.encoder_input_placeholders,
                self.decoder_input_placeholders,
                stacked_lstms,
                self.xvocab_size,
                self.yvocab_size,
                self.emb_dim,
                feed_previous=True
            )

    def build_rnn_model(self, stacked_lstms):
        # Sharing of parameters between training and testing models
        with tf.variable_scope('encdec') as scope:
            self.decode_outputs, self.decode_states, self.context_vector_output = embedding_rnn_seq2seq(
                self.encoder_input_placeholders,
                self.decoder_input_placeholders,
                stacked_lstms,
                self.xvocab_size,
                self.yvocab_size,
                self.emb_dim
            )

            scope.reuse_variables()

            # Testing model. Here the output from the previous timestep is fed as input to the next timestep
            self.decode_outputs_test, self.decode_states_test, self.context_vector_output_test = embedding_rnn_seq2seq(
                self.encoder_input_placeholders,
                self.decoder_input_placeholders,
                stacked_lstms,
                self.xvocab_size,
                self.yvocab_size,
                self.emb_dim,
                feed_previous=True
            )
