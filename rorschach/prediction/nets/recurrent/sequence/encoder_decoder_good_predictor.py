#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import (GRU, Activation, Dense, Dropout, Masking, Merge,
                          TimeDistributed, LSTM, Permute, Bidirectional, Embedding, Input)
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


class EncoderDecoderGoodPredictor(BasePredictor):

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

        inputs = []
        encoders = []
        encoder_hidden = []

        for i in range(5):
            # Input
            current_input = Input(
                shape=(10,)
            )

            # Embedding
            current_embedding = Embedding(
                300,
                19,
                mask_zero=False
            )(current_input)

            # Encoder
            current_encoder = None
            current_encoder_hidden = None

            if i == 0:
                # First input, has only one input
                current_encoder, *current_encoder_hidden = HiddenStateLSTM(
                    512,
                    dropout_W=0.5,
                    # dropout_U=0.5,
                    #W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                    #b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
                    return_sequences=False
                )(current_embedding)
            else:
                if i == 4:
                    # Input 5, has two inputs like the rest, but also returns the entire sequence
                    current_encoder, _, _ = HiddenStateLSTM(
                        512,
                        dropout_W=0.5,
                        #dropout_U=0.5,
                        #W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                        #b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
                        return_sequences=True
                    )([current_embedding, encoder_hidden[-1][-1], encoders[-1]])

                else:
                    # Input 2 - 4 has two inputs, the input and the previous LSTM
                    current_encoder, *current_encoder_hidden = HiddenStateLSTM(
                        512,
                        dropout_W=0.5,
                        #dropout_U=0.5,
                        #W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                        #b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
                        return_sequences=False
                    )([current_embedding, encoder_hidden[-1][-1], encoders[-1]])

            inputs.append(current_input)
            encoders.append(current_encoder)

            if current_encoder_hidden is not None:
                encoder_hidden.append(current_encoder_hidden)

        #
        # DECODER PART GOES HERE
        #

        decoders = []
        decoder_hidden = []
        outputs = []

        for i in range(10):
            # Decoder
            current_decoder = None
            current_decoder_hidden = None

            if i == 0:
                # Input the output of the encoders
                current_decoder, *current_decoder_hidden = HiddenStateLSTM(
                    512,
                    dropout_W=0.5,
                    #dropout_U=0.5,
                    #W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                    #b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
                    return_sequences=False
                )([])
            else:
                current_decoder, *current_decoder_hidden = HiddenStateLSTM(
                    512,
                    dropout_W=0.5,
                    #dropout_U=0.5,
                    #W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                    #b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
                    return_sequences=False
                )([encoders[-1], decoders[-1], decoder_hidden[-1][-1]])

            # Output
            current_output = Dense(19)(current_decoder)
            current_output = Activation('softmax')(current_output)

            decoders.append(current_decoder)
            decoder_hidden.append(current_decoder_hidden)
            outputs.append(current_output)

        # Optimizer
        sgd = SGD(lr=1., decay=1e-6, momentum=0.9, nesterov=True)

        # Compile model
        self.model = Model(input=inputs, output=outputs)
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

