#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import (GRU, Activation, Dense, Dropout, Masking, Merge,
                          TimeDistributed)
from keras.layers.convolutional import AveragePooling1D
from keras.models import Sequential
from keras.utils.visualize_util import plot

from rorschach.prediction.callbacks import PlotCallback
from rorschach.prediction.helpers import (EmbeddingCalculator,
                                          WidthCalculator)
from rorschach.prediction.nets import BasePredictor
from rorschach.utilities import Config, Filesystem, LoggerWrapper, unpickle_data  # isort:skip


class SequencePredictor(BasePredictor):

    def __init__(self):
        super().__init__()

        self.log = LoggerWrapper.load(__name__)
        self.model = None
        self.callback = None

    def prepare(self):
        self.widest = WidthCalculator.calc(self.training_images_transformed)
        self.embeddings = EmbeddingCalculator.calc(self.training_labels_transformed)
        self.pooling_factor = unpickle_data(Filesystem.get_root_path('data/pooling.pickl'))['factor']

        self.keras_setup()

    def keras_setup(self):
        self.callback = PlotCallback()
        self.callback.epochs = Config.get('predicting.epochs')


        # print(self.embedding_values_translated)
        # print(self.training_images_transformed.shape)
        # self.model.add(LSTM(len(self.embedding_nums), return_sequences=True, stateful=True,
        #               batch_input_shape=(Config.get('predicting.batch_size'), 1, self.widest)))
        # self.model.add(LSTM(32, return_sequences=True, stateful=True))
        # self.model.add(AttentionSeq2Seq(output_dim=len(self.embedding_nums),
        #                                     hidden_dim=len(self.embedding_nums),
        #                      input_dim=self.widest,
        #                      output_length=self.widest,
        #                      depth=4
        #                      ))

        left = Sequential()
        left.add(Masking(mask_value=0.,
                         input_shape=(self.widest, 1)))
        left.add(GRU(output_dim=256,
                     activation='sigmoid',
                     inner_activation='hard_sigmoid',
                     return_sequences=True))

        right = Sequential()
        right.add(Masking(mask_value=0.,
                          input_shape=(self.widest, 1)))
        right.add(GRU(output_dim=256,
                      activation='sigmoid',
                      inner_activation='hard_sigmoid',
                      return_sequences=True))

        self.model = Sequential()
        self.model.add(Merge([left, right],
                             mode='concat'))

        # self.model.add(Embedding(self.voc_size + 1,
        #                          256,
        #                          input_length=self.widest))
        # self.model.add(Masking(mask_value=0., input_shape=(self.widest, 1)))
        # self.model.add(Masking(mask_value=0., input_shape=(self.widest, 1)))
        # self.model.add(GRU(output_dim=256,
        #                     activation='sigmoid',
        #                     inner_activation='hard_sigmoid',
        #                     return_sequences=True))

        self.model.add((GRU(256,
                            return_sequences=True)))
        self.model.add((GRU(128,
                            return_sequences=True)))
        self.model.add(Dropout(0.2))
        self.model.add(TimeDistributed(Dense(self.embeddings)))
        self.model.add(AveragePooling1D(pool_length=self.pooling_factor))
        # self.model.add(TimeDistributed(Dense(len(self.embedding_nums))))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        self.model.summary()

        plot(self.model, to_file='model_rnn.png', show_shapes=True)

    def train(self):
        self.log.info('Begin training')
        print(self.training_images_transformed.shape)
        print(self.training_labels_transformed.shape)

        self.model.fit([self.training_images_transformed,
                        self.training_images_transformed
                        ],
                       self.training_labels_transformed,
                       nb_epoch=Config.get('predicting.epochs'),
                       verbose=1,
                       batch_size=Config.get('predicting.batch_size'),
                       validation_split=0.2,
                       shuffle=True,
                       callbacks=[self.callback]
                       )

        self.log.info('Finished training')

    def predict(self):
        self.log.info('Begin predicting')

        self.log.info('Finished')
