#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import (GRU, Activation, Dense, Dropout, Masking, Merge,
                          TimeDistributed, LSTM, Permute, Bidirectional, InputLayer, Input)
from keras.layers.convolutional import AveragePooling1D
from keras.regularizers import WeightRegularizer, ActivityRegularizer
from keras.models import Sequential, Model
from keras.utils.visualize_util import plot

from rorschach.prediction.callbacks import CallbackWrapper
from rorschach.prediction.callbacks.plotter import PlotCallback
from rorschach.prediction.helpers import (EmbeddingCalculator,
                                          WidthCalculator)
from rorschach.prediction.layer import Attention, AttentionLSTM
from rorschach.prediction.nets import BasePredictor
from rorschach.utilities import Config, Filesystem, LoggerWrapper, unpickle_data  # isort:skip


class AttentionLayerSequencePredictor2(BasePredictor):

    def __init__(self):
        super().__init__()

        self.log = LoggerWrapper.load(__name__)
        self.model = None
        self.callback = None

    def prepare(self):
        self.keras_setup()

    def keras_setup(self):
        #self.callback = PlotCallback()
        #self.callback.epochs = Config.get('predicting.epochs')

        '''
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(30, 4)))
        self.model.add(Masking(mask_value=0))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Attention())

        self.model.add(Dense(output_dim=10))
        '''
        input = Input(shape=(30, 4))
        activations = LSTM(64, return_sequences=True)(input)
        sentence = Attention()(activations)

        probabilities = Dense(10,)(sentence)
        probabilities = Activation('linear')(probabilities)

        self.model = Model(input=input, output=probabilities)
        self.model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

        #self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        self.model.summary()

        plot(self.model, to_file='model_rnn.png', show_shapes=True)

    def train(self):
        self.log.info('Begin training')
        self.model.fit(self.training_images_transformed,
                       self.training_labels_transformed,
                       nb_epoch=Config.get('predicting.epochs'),
                       verbose=1,
                       batch_size=Config.get('predicting.batch_size'),
                       validation_data=(self.test_images_transformed, self.test_labels_transformed),
                       #callbacks=[self.callback]
                       )

        self.log.info('Finished training')

    def predict(self):
        self.log.info('Begin predicting')

        derp = self.model.predict(self.training_images_transformed)[0]
        print(list(derp[0:20]))

