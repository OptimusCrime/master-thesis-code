#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, GRU, Reshape, Activation, Embedding, TimeDistributed, Dense
from keras.utils.visualize_util import plot
import sys

from nets.base import BasePredictor
from utilities import Config, LoggerWrapper


class RNNPredictor(BasePredictor):

    def __init__(self):
        super().__init__()

        self.log = LoggerWrapper.load(__name__)
        self.model = None

        # Storing the transformed data
        self.training_images_transformed = None
        self.training_labels_transformed = None
        self.phrase_transformed = None

    def preprocess(self):
        self.transform_data_set()
        self.transform_phrase()

        self.keras_setup()

    def transform_data_set(self):
        local_training_images = []
        local_training_labels = []

        for data in self.data_set:
            local_training_images.append(data['matrix'][0])
            local_training_labels.append(data['label'])

        self.training_images_transformed = np.array(local_training_images)
        self.training_labels_transformed = np.array(local_training_labels)

    def transform_phrase(self):
        self.phrase_transformed = self.phrase[0]['matrix'][0]

    def keras_setup(self):
        self.model = Sequential()
        self.model.add(Embedding(1000, 128, dropout=0.2, name="embedding_1"))
        self.model.add(LSTM(len(Config.get('general.characters')), dropout_W=0.2, dropout_U=0.2, name="lstm_1", return_sequences=True))
        #self.model.add(LSTM(26, dropout_W=0.2, dropout_U=0.2, name="lstm_1", input_shape=(None, 1)))
        #self.model.add(Embedding(256, 26, dropout=0.2, name="embedding_2"))
        #self.model.add(Dense(len(Config.get('general.characters')), name="dense_1"))
        #self.model.add(Reshape((26,), input_shape=(26,)))
        self.model.add(Activation('softmax', name="activation_1"))
        #self.model.add(TimeDistributed(Dense(1, activation='softmax')))

        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        self.model.summary()

        #plot(self.model, to_file='model_rnn.png', show_shapes=True)

    def train(self):
        self.log.info('Begin training')

        self.model.fit(self.training_images_transformed,
                       self.training_labels_transformed,
                       nb_epoch=5,
                       verbose=1,
                       batch_size=40
                       )

        self.log.info('Finished training')

    def predict(self):
        self.log.info('Begin predicting')

        predictions = self.model.predict(self.phrase_transformed)

        for line in predictions:
            print(line[0])

        self.log.info('Finished')


