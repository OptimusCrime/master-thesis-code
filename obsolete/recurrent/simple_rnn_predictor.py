#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, GRU, Reshape, Activation, Embedding, TimeDistributed, Dense, Input, Layer, \
    Dropout
from keras.layers.wrappers import Bidirectional
from keras.optimizers import SGD
from keras.utils.visualize_util import plot

from rorchach.nets.base import BasePredictor
from rorchach.utilities import Config, LoggerWrapper, MatrixDim


class SimpleRNNPredictor(BasePredictor):

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

        #self.training_images_transformed = self.training_images_transformed.reshape(
        #    len(self.training_images_transformed),
        #    self.training_images_transformed.shape[1],
        #    1
        #)

    def transform_phrase(self):
        self.phrase_transformed = self.phrase[0]['matrix'][0]

        #self.phrase_transformed = self.phrase_transformed.reshape(
        #    1,
        #    self.phrase_transformed.shape[0],
        #    1
        #)

        #self.phrase_transformed = self.phrase_transformed.reshape(
        #    1,
        #    self.phrase_transformed.shape[0],
        #    len(Config.get('general.characters'))
        #)

    def keras_setup(self):
        #print(self.training_images_transformed.shape)
        self.model = Sequential()
        self.model.add(Embedding(MatrixDim.get_size(), 52, dropout=0.0, name="embedding_2"))


        self.model.add(Bidirectional(
            LSTM(52,
                 return_sequences=True),
            #input_shape=(
            #    self.training_images_transformed.shape[1],
            #    self.training_images_transformed.shape[2]
            #)
            ))

        self.model.add(Bidirectional(LSTM(MatrixDim.get_size(),
                                          return_sequences=True)))
        #self.model.add(LSTM(26, dropout_W=0.2, dropout_U=0.2, name="lstm_1", input_shape=(None, 1)))
        #
        #self.model.add(Dense(len(Config.get('general.characters')), name="dense_1"))
        #self.model.add(Reshape((26,), input_shape=(26,)))
        self.model.add(TimeDistributed(Dense(MatrixDim.get_size())))
        self.model.add(Dropout(0.2))
        self.model.add(Activation('softmax',
                                  name="activation_1"))
        #self.model.add(TimeDistributed(Dense(1, activation='softmax')))

        # Compile with sgd
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.model.summary()

        plot(self.model, to_file='model_rnn.png', show_shapes=True)

    def train(self):
        self.log.info('Begin training')

        self.model.fit(self.training_images_transformed,
                       self.training_labels_transformed,
                       nb_epoch=Config.get('predicting.epochs'),
                       verbose=1,
                       batch_size=Config.get('predicting.batch_size')
                       )

        self.log.info('Finished training')

    def predict(self):
        self.log.info('Begin predicting')

        predictions = self.model.predict(self.phrase_transformed)

        for line in predictions:
            print(line)

        self.log.info('Finished')


