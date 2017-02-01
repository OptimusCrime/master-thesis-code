#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import copy

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import (GRU, Activation, Dense, Dropout, Masking, Merge,
                          TimeDistributed, LSTM, Permute, Bidirectional, Embedding, Input, Reshape, merge)
from keras.layers.convolutional import AveragePooling1D
from keras.regularizers import WeightRegularizer, ActivityRegularizer
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.utils.visualize_util import plot

from rorschach.prediction.callbacks import CallbackWrapper
from rorschach.prediction.callbacks.plotter import PlotCallback
from rorschach.prediction.helpers import (EmbeddingCalculator,
                                          WidthCalculator)
from rorschach.prediction.layer import HiddenStateLSTM
from rorschach.prediction.nets import BasePredictor
from rorschach.utilities import Config, Filesystem, LoggerWrapper, unpickle_data  # isort:skip


class EncoderDecoderBestPredictor(BasePredictor):

    def __init__(self):
        super().__init__()

        self.log = LoggerWrapper.load(__name__)
        self.model = None
        self.callback = None

    def prepare(self):
        self.callback_setup()
        self.keras_setup()

    def callback_setup(self):
        self.callback = CallbackWrapper([PlotCallback])
        self.callback.data['epochs'] = Config.get('predicting.epochs')

    def keras_setup(self):

        #
        # ENCODER PART GOES HERE
        #

        input_layer = Input(
            shape=(48,)
        )

        # Embedding
        embedding_layer = Embedding(
            300,
            19,
            mask_zero=False
        )(input_layer)

        # First input, has only one input
        encoder, *encoder_hidden_states = HiddenStateLSTM(
            128,
            dropout_W=0.2,
            dropout_U=0.2,
            W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
            b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
            return_sequences=False
        )(embedding_layer)

        context = Reshape((1, 128))(encoder)

        hidden_state1 = copy(encoder_hidden_states[0])
        hidden_state2 = copy(encoder_hidden_states[1])

        outputs = []
        decoders = []

        for i in range(10):
            # Decoder
            current_decoder = None
            current_decoder_hidden = None

            if i == 0:
                # Input the output of the encoders
                current_decoder, *current_decoder_hidden = HiddenStateLSTM(
                    128,
                    dropout_W=0.2,
                    dropout_U=0.2,
                    W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                    b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
                    return_sequences=False
                )([context, hidden_state1, hidden_state2])
            else:
                previous_output = Reshape((1, 128))(decoders[-1])

                print('context')
                print(context)
                print('previous')
                print(previous_output)

                ouput_merge = merge([context, previous_output])

                current_decoder, *current_decoder_hidden = HiddenStateLSTM(
                    128,
                    dropout_W=0.2,
                    dropout_U=0.2,
                    W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                    b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
                    return_sequences=False
                )([ouput_merge, hidden_state1, hidden_state2])

            # Output
            current_output = Dense(19)(current_decoder)
            current_output = Activation('softmax')(current_output)

            decoders.append(current_decoder)
            outputs.append(current_output)

        # Optimizer
        sgd = SGD(lr=1., decay=1e-6, momentum=0.9, nesterov=True)

        # Compile model
        self.model = Model(input=input_layer, output=outputs)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # Print summary
        self.model.summary()

        # Save model plot
        plot(self.model, to_file='test1232.png', show_shapes=True)

    def train(self):
        self.log.info('Begin training')

        self.model.fit(
            self.training_images_transformed,
            self.training_labels_transformed,
            nb_epoch=Config.get('predicting.epochs'),
            verbose=1,
            batch_size=Config.get('predicting.batch_size'),
            validation_data=(self.test_images_transformed, self.test_labels_transformed),
            callbacks=[self.callback]
        )

        self.log.info('Finished training')

    def predict(self):
        self.log.info('Begin predicting')

        derp = self.model.predict(
            self.test_images_transformed,
            batch_size=100
        )

        print(list(derp[0:2]))

