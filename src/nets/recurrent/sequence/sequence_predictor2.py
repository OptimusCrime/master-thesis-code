#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, GRU, Reshape, Activation, Embedding, TimeDistributed, Dense, Dropout, Masking
from keras.layers.convolutional import AveragePooling1D
from keras.optimizers import SGD
from keras.utils.visualize_util import plot

from seq2seq.models import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq

from nets.base import BasePredictor
from utilities import Config, LoggerWrapper, MatrixDim, Filesystem, unpickle_data


class SequencePredictor2(BasePredictor):

    def __init__(self):
        super().__init__()

        self.log = LoggerWrapper.load(__name__)
        self.model = None
        self.widest = None

        # Storing the transformed data
        self.embedding_values = {}
        self.embedding_values_translated = {}
        self.embedding_nums = set()
        self.translations = {}
        self.voc_size = None
        self.training_images_transformed = None
        self.training_labels_transformed = None
        self.phrase_transformed = None
        self.pooling_factor = None
        self.labels_width = None

    def preprocess(self):
        print(self.data_set)
        self.embedding_stuff()
        self.embedding_counts()

        self.transform_data_set()
        self.transform_phrase()

        self.keras_setup()

    def embedding_stuff(self):
        self.embedding_values = unpickle_data(Filesystem.get_root_path('data/unique_signatures_data.pickl'))

        print('self.embedding_values 1')
        print(self.embedding_values)

        for key in self.embedding_values:
            #self.embedding_values[key] += 0
            self.embedding_values[key] += 1

        print('self.embedding_values 2')
        print(self.embedding_values)

        for key in self.embedding_values:
            self.embedding_nums.add(self.embedding_values[key])

            if self.embedding_values[key] not in self.embedding_values_translated:
                self.embedding_values_translated[self.embedding_values[key]] = [key]
            else:
                self.embedding_values_translated[self.embedding_values[key]].append(key)

        print('self.embedding_values_translated')
        print(self.embedding_values_translated)
        self.embedding_values_translated[0] = '-'

    def embedding_counts(self):
        uniques = {}
        for set in [self.data_set, self.phrase]:
            for data in set:
                for v in range(len(data['embedding_raw']) - 2):
                    value = data['embedding_raw'][v]
                    if value not in uniques:
                        uniques[value] = 1
                    else:
                        uniques[value] += 1

        print('uniques')
        print(uniques)

        # DERP
        self.voc_size = len(uniques)

        # We are now sorting the values descending by their popularity
        sorted_uniques_values = sorted(uniques, key=uniques.get, reverse=True)

        print('sorted_uniques_values')
        print(sorted_uniques_values)

        # Reassign the dict's values to be the index from the sorting, e.i. their popularity. The list still has a
        # access complexity of O(1) which is what we want in the next step.
        for i in range(len(sorted_uniques_values)):
            self.translations[sorted_uniques_values[i]] = i + 1
            #self.translations[sorted_uniques_values[i]] = i

        print('self.translations')
        print(self.translations)


    def transform_data_set(self):
        for sets in [self.data_set, self.phrase]:
            for data in sets:
                if self.widest is None or len(data['embedding_raw']) > self.widest:
                    self.widest = len(data['embedding_raw'])

        self.training_images_transformed = self.transform_set(self.data_set)
        #self.training_labels_transformed = np.zeros((len(self.data_set), self.widest, len(self.embedding_nums)),
        #                                            dtype=np.float32)
        #print(self.training_labels_transformed.shape)
        #self.training_labels_transformed = np.zeros((len(self.data_set), self.widest, len(self.embedding_nums) + 1),
        #                                            dtype=np.float32)

        longest_word = None
        for i in range(len(self.data_set)):
            if longest_word is None or len(self.data_set[i]['text']) > longest_word:
                longest_word = len(self.data_set[i]['text'])

        div_facors = set()
        for div in [2, 3]:
            valids = [self.widest]
            while True:
                calc = valids[-1] / float(div)
                print(calc)
                if calc.is_integer():
                    valids.append(int(calc))
                    if calc < longest_word:
                        del valids[-1]
                        break

                div_facors.update(valids)

                break


        self.pooling_factor = min(div_facors)
        self.labels_width = self.widest // self.pooling_factor

        self.training_labels_transformed = np.zeros((len(self.data_set), self.labels_width, len(self.embedding_nums) + 1),
                                                    dtype=np.float32)

        for i in range(len(self.data_set)):
            for j in range(self.labels_width):
                if j < len(self.data_set[i]['text']):
                    char = self.data_set[i]['text'][j]
                    #print(char)
                    #print(self.embedding_values[char])
                    self.training_labels_transformed[i][j][self.embedding_values[char]] = 1.
                else:
                    self.training_labels_transformed[i][j][0] = 1.

        #for i in range(len(self.training_labels_transformed)):
        #    print(self.training_labels_transformed[i])


    def transform_phrase(self):
        self.phrase_transformed = self.transform_set(self.phrase)

    def transform_set(self, data_set):
        transform_set = np.zeros((len(data_set), self.widest), dtype=np.int32)

        for i in range(len(data_set)):
            for v in range(len(data_set[i]['embedding_raw']) - 2):
                val = data_set[i]['embedding_raw'][v]
                if v != '0':
                    transform_set[i][v] = self.translations[val]

        return transform_set

    def keras_setup(self):
        #print(self.embedding_values_translated)
        '''print(self.training_images_transformed.shape)



        self.model.add(LSTM(len(self.embedding_nums), return_sequences=True, stateful=True,
                       batch_input_shape=(Config.get('predicting.batch_size'), 1, self.widest)))
        #self.model.add(LSTM(32, return_sequences=True, stateful=True))
        '''

        #self.model.add(AttentionSeq2Seq(output_dim=len(self.embedding_nums),
        #                                    hidden_dim=len(self.embedding_nums),
        #                     input_dim=self.widest,
        #                     output_length=self.widest,
        #                     depth=4
        #                     ))

        '''
        FOR NON SIMPLE:
        self.model.add(SimpleSeq2Seq(output_dim=len(self.embedding_nums),
                             hidden_dim=500,
                             input_dim=self.widest,
                             output_length=self.widest,
                             depth=4
                             ))

        '''

        # Working seq2seq with terrible results goes here
        #self.model = Sequential()
        #self.model.add(Seq2Seq(output_dim=len(self.embedding_nums) + 1,
        #        hidden_dim=500,
        #        input_dim=1,
        #        output_length=self.widest,
        #        depth=4
        #        ))
        #
        #self.model.add(Activation('softmax',
        #                          name="activation_1"))

        print(len(self.embedding_nums) + 1)

        self.model = Sequential()
        self.model.add(Embedding(self.voc_size + 1, 256, input_length=self.widest))
        #self.model.add(Masking(mask_value=0., input_shape=(self.widest, 1)))
        self.model.add(GRU(output_dim=256, activation='sigmoid',
                            inner_activation='hard_sigmoid',
                            return_sequences=True))
        self.model.add(GRU(256, return_sequences=True))  # returns a sequence of vectors of dimension 32
        self.model.add(GRU(128, return_sequences=True))  # return a single vector of dimension 32
        self.model.add(Dropout(0.2))
        self.model.add(TimeDistributed(Dense(len(self.embedding_nums) + 1)))
        self.model.add(AveragePooling1D(pool_length=self.pooling_factor))
        #self.model.add(TimeDistributed(Dense(len(self.embedding_nums))))
        self.model.add(Activation('softmax'))

        # self.model.add(TimeDistributed(Dense(1, activation='softmax')))

        # Compile with sgd
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
        #self.model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
        #self.model.compile(loss='mse', optimizer='rmsprop')

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        #self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        #self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model.summary()

        plot(self.model, to_file='model_rnn.png', show_shapes=True)

    def train(self):
        self.log.info('Begin training')
        print(self.training_images_transformed.shape)
        print(self.training_labels_transformed.shape)

        self.model.fit(self.training_images_transformed,
                       self.training_labels_transformed,
                       nb_epoch=Config.get('predicting.epochs'),
                       verbose=1,
                       batch_size=Config.get('predicting.batch_size'),
                       validation_split=0.2,
                       shuffle=True
                       )

        self.log.info('Finished training')

    def predict(self):
        self.log.info('Begin predicting')

        print(self.phrase_transformed)

        predictions = self.model.predict(self.phrase_transformed)

        print(predictions)
        print(predictions.shape)
        for line in predictions[0]:
            idx = np.argmax(line)
            print(line)
            print(idx)
            print(line[idx])
            print(self.embedding_values_translated[idx])
            print('----')


        self.log.info('Finished')

