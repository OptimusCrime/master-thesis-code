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
        self.keras_setup()

    def keras_setup(self):
        self.callback = CallbackWrapper([PlotCallback])
        self.callback.data['epochs'] = Config.get('predicting.epochs')

        '''for derp in [self.training_images_transformed, self.training_labels_transformed]:
            for i in range(len(derp)):
                print(derp[i][6])
            print('=====================================================================')
        '''
        inputs = []
        encoders = []
        hidden = []

        for i in range(5):
            name_input = 'in' + str(i)
            embedding_name = 'embedding' + str(i)
            encoder_intermediate = 'encoder' + str(i)

            current_input = Input(
                shape=(10,),
                name=name_input)
            current_embedding = Embedding(300,
                                          19,
                                          mask_zero=True,
                                          name=embedding_name)(current_input)

            current_encoder = None
            current_hidden = None
            if i == 0:
                # First input, has only one input
                current_encoder, *current_hidden = HiddenStateLSTM(256,
                                                                   dropout_W=0.5,
                                                                   dropout_U=0.5,
                                                                   W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                                                                   b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
                                                                   return_sequences=False)(current_embedding)
            else:
                if i == 4:
                    # Input 5, has two inputs like the rest, but also returns the entire sequence
                    current_encoder, _, _ = HiddenStateLSTM(256,
                                                            dropout_W=0.5,
                                                            dropout_U=0.5,
                                                            W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                                                            b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
                                                            return_sequences=True)([current_embedding] + hidden[-1])

                else:
                    # Input 2 - 4 has two inputs, the input and the previous LSTM
                    current_encoder, *current_hidden = HiddenStateLSTM(
                        256,
                        dropout_W=0.5,
                        dropout_U=0.5,
                        W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                        b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
                        return_sequences=False)([current_embedding] + hidden[-1])

            inputs.append(current_input)
            encoders.append(current_encoder)

            if current_hidden is not None:
                hidden.append(current_hidden)

        decoders = []
        decoder_hidden = []
        decoder_inner = []
        outputs = []
        for i in range(10):
            if i == 0:
                # Input the output of the encoders
                current_decoder, *current_decoder_hidden = HiddenStateLSTM(
                    256,
                    dropout_W=0.5,
                    dropout_U=0.5,
                    W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                    b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
                    return_sequences=True)(encoders[-1])
            else:
                current_decoder, *current_decoder_hidden = HiddenStateLSTM(
                    256,
                    dropout_W=0.5,
                    dropout_U=0.5,
                    W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                    b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
                    return_sequences=True)([decoders[-1]] + decoder_hidden[-1])

            current_inner_decoder = LSTM(
                256,
                dropout_W=0.5,
                dropout_U=0.5,
                W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
                return_sequences=False)(current_decoder)

            current_output = Dense(19)(current_inner_decoder)
            current_output = Activation('softmax')(current_output)

            decoders.append(current_decoder)
            decoder_hidden.append(current_decoder_hidden)
            decoder_inner.append(current_inner_decoder)
            outputs.append(current_output)

        sgd = SGD(lr=1., decay=1e-6, momentum=0.9, nesterov=True)

        self.model = Model(input=inputs, output=outputs)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        self.model.summary()

        plot(self.model, to_file='test1232.png', show_shapes=True)

    def train(self):
        self.log.info('Begin training')
        self.model.fit(self.training_images_transformed,
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

        derp = self.model.predict(self.test_images_transformed, batch_size=100)
        print(list(derp[0:2]))

