# -*- coding: utf-8 -*-

import tensorflow as tf

from rorschach.prediction.tensorflow.layers import AbstractSeq2seq
from rorschach.utilities import Config


class AttentionSeq2Seq(AbstractSeq2seq):

    def build_model(self):
        rnn_cell = tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.LSTMCell(
                self.emb_dim,
                state_is_tuple=True
            ),
            output_keep_prob=self.keep_probability
        )

        # Stacked LSTMs (defined by the number of layers in the model)
        stacked_lstms = tf.contrib.rnn.MultiRNNCell(
            [rnn_cell] * self.num_layers,
            state_is_tuple=True
        )

        # Sharing of parameters between training and testing models
        with tf.variable_scope('decoder') as scope:
            self.decode_outputs, self.decode_states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                self.encoder_input_placeholders,
                self.decoder_input_placeholders,
                stacked_lstms,
                self.xvocab_size,
                self.yvocab_size,
                self.emb_dim
            )

            scope.reuse_variables()

            # Testing model. Here the output from the previous timestep is fed as input to the next timestep
            self.decode_outputs_test, self.decode_states_test = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                self.encoder_input_placeholders,
                self.decoder_input_placeholders,
                stacked_lstms,
                self.xvocab_size,
                self.yvocab_size,
                self.emb_dim,
                feed_previous=True
            )
