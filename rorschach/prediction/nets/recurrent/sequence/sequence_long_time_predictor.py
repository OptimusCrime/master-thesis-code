#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import (GRU, Activation, Dense, Dropout, Masking, Merge,
                          TimeDistributed, LSTM, Permute, Bidirectional)
from keras.layers.convolutional import AveragePooling1D
from keras.regularizers import WeightRegularizer, ActivityRegularizer
from keras.models import Sequential
from keras.utils.visualize_util import plot

from rorschach.prediction.callbacks import CallbackWrapper
from rorschach.prediction.callbacks.plotter import PlotCallback
from rorschach.prediction.helpers import (EmbeddingCalculator,
                                          WidthCalculator)
from rorschach.prediction.nets import BasePredictor
from rorschach.utilities import Config, Filesystem, LoggerWrapper, unpickle_data  # isort:skip


class SequenceLongTimePredictor(BasePredictor):

    def __init__(self):
        super().__init__()

        self.log = LoggerWrapper.load(__name__)
        self.model = None
        self.callback = None

    def prepare(self):
        print('hello1')
        self.keras_setup()

    def keras_setup(self):
        print(self.training_images_transformed.shape)
        self.callback = PlotCallback()
        self.callback.epochs = Config.get('predicting.epochs')

        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(output_dim=256,
                                          return_sequences=True,
                                          W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                                          b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01)),
                                     input_shape=(100, 1)))

        self.model.add(Dropout(0.4))
        self.model.add(Bidirectional(LSTM(128,
                                          return_sequences=True,
                                          W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                                          b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01))))
        self.model.add(Dropout(0.4))

        self.model.add(TimeDistributed(Dense(output_dim=19)))

        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

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
                       callbacks=[self.callback]
                       )

        self.log.info('Finished training')

    def predict(self):
        self.log.info('Begin predicting')

        derp = self.model.predict(self.training_images_transformed)[0]
        print(list(derp[0:20]))

