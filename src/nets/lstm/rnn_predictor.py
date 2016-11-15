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
        self.keras_setup()

    def transform_data_set(self):
        local_training_images = []
        local_training_labels = []

        for data in self.data_set:
            local_training_images.append(data['matrix'][0])
            local_training_labels.append(data['label'])

        self.training_images_transformed = np.array(local_training_images)
        self.training_labels_transformed = np.array(local_training_labels)

    def keras_setup(self):
        self.model = Sequential()
        self.model.add(Embedding(1000, 128, dropout=0.2, name="embedding_1"))
        self.model.add(LSTM(len(Config.get('general.characters')), dropout_W=0.2, dropout_U=0.2, name="lstm_1", return_sequences=True))
        #self.model.add(LSTM(26, dropout_W=0.2, dropout_U=0.2, name="lstm_1", input_shape=(None, 1)))
        #self.model.add(Embedding(256, 26, dropout=0.2, name="embedding_2"))
        #self.model.add(Dense(len(Config.get('general.characters')), name="dense_1"))
        #self.model.add(Reshape((26,), input_shape=(26,)))
        self.model.add(Activation('sigmoid', name="activation_1"))
        #self.model.add(TimeDistributed(Dense(1, activation='softmax')))

        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        self.model.summary()

        plot(self.model, to_file='model_rnn.png', show_shapes=True)

    def train(self):
        image_set_size = len(self.training_images_transformed)
        for n in range(Config.get('predicting.epochs')):
            self.log.info('=============== Training epoch %s ===============', n + 1)
            for i in range(image_set_size):
                history = self.model.fit(np.array([self.training_images_transformed[i]]),
                               np.array([self.training_labels_transformed[i]]),
                               nb_epoch=1,
                               verbose=0,
                               batch_size=1
                               )

                if (i + 1) % Config.get('logging.batch_reporting') == 0:
                    self.log.info('Image %s/%s. Loss = %.4f, acc = %.4f', i + 1, image_set_size,
                                  history.history['loss'][0], history.history['acc'][0])




    def predict(self):
        predictions = self.model.predict(self.training_images_transformed[0])
        print(np.array_str(predictions, max_line_width=300, precision=3))
