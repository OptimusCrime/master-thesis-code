#!/usr/bin/env python
# -*- coding: utf-8 -*-

import operator
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, GRU, Reshape, Activation, Embedding, TimeDistributed, Dense
from keras.optimizers import SGD
from keras.utils.visualize_util import plot

from rorchach.prediction.base import BasePredictor
from rorchach.utilities import Config, LoggerWrapper, MatrixDim


class EmbeddingRNNPredictor(BasePredictor):

    def __init__(self):
        super().__init__()

        self.log = LoggerWrapper.load(__name__)
        self.model = None

        # Storing the transformed data
        self.embedding_values = {}
        self.training_images_transformed = None
        self.training_labels_transformed = None
        self.phrase_transformed = None

    def preprocess(self):
        self.calculate_embedding_pos()
        self.transform_data_set()
        self.transform_phrase()

        self.keras_setup()

    def transform_data_set(self):
        local_training_images = []
        local_training_labels = []

        self.log.info('Constructing input vectors with embedded values.')
        for data in self.data_set:
            local_training_images.append(self.rich_values_to_embedding(data['embedding']))
            local_training_labels.append(data['label'])

        self.training_images_transformed = np.array(local_training_images)
        self.training_labels_transformed = np.array(local_training_labels)

        self.log.info('Finished transforming data sets.')

    def calculate_embedding_pos(self):
        # Loop the data sets. We need to loop the phrase, in the rare case that the phrase contains some unique
        # values that are NOT in the data set. This could potentially crash the algorithm.
        self.log.info('Translating input values to embedding values.')
        for data_set in [self.data_set, self.phrase]:
            for data in data_set:
                for v in data['embedding']:
                    if v not in self.embedding_values:
                        self.embedding_values[v] = 1
                    else:
                        self.embedding_values[v] += 1

        # We are now sorting the values descending by their popularity
        sorted_embedding_values = sorted(self.embedding_values, key=self.embedding_values.get, reverse=True)

        # Reassign the dict's values to be the index from the sorting, e.i. their popularity. The list still has a
        # access complexity of O(1) which is what we want in the next step.
        for i in range(len(sorted_embedding_values)):
            self.embedding_values[sorted_embedding_values[i]] = i

    def rich_values_to_embedding(self, vector):
        embedding_matrix = np.zeros_like(vector, dtype=np.int)
        for i in range(len(vector)):
            embedding_matrix[i] = self.embedding_values[vector[i]]

        return embedding_matrix

    def transform_phrase(self):
        self.phrase_transformed = self.rich_values_to_embedding(self.phrase[0]['embedding'])

    def keras_setup(self):
        self.model = Sequential()
        self.model.add(Embedding(len(self.embedding_values) + 1,
                                 128,
                                 dropout=0,
                                 name="embedding_1"
                                 ))

        self.model.add(LSTM(MatrixDim.get_size(), dropout_W=0.2, dropout_U=0.2, name="lstm_1", return_sequences=True))
        self.model.add(Activation('softmax', name="activation_1"))

        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
        #self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        self.model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

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

        print(self.phrase_transformed)
        print(self.phrase[0]['embedding'])

        predictions = self.model.predict(self.phrase_transformed)

        for line in predictions:
            print(line[0])

        self.log.info('Finished')


