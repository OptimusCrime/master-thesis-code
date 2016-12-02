#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, GRU, Reshape, Activation, Embedding, TimeDistributed, Dense
from keras.optimizers import SGD
from keras.utils.visualize_util import plot
import seq2seq
from seq2seq.models import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq


from nets.base import BasePredictor
from utilities import Config, LoggerWrapper, MatrixDim


class AttentionSeqPredictor(BasePredictor):

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

        self.training_images_transformed = self.training_images_transformed.reshape(
            len(self.training_images_transformed),
            1,
            self.training_images_transformed.shape[1]

        )

        print(self.training_labels_transformed.shape)

    def transform_phrase(self):
        self.phrase_transformed = self.phrase[0]['matrix'][0]
        self.phrase_transformed = self.phrase_transformed.reshape(
            1, 1, len(self.phrase[0]['matrix'][0]))


    def keras_setup(self):
        #self.model = Sequential()
        self.model = AttentionSeq2Seq(output_dim=MatrixDim.get_size(),
                             hidden_dim=500,
                             input_dim=len(self.phrase[0]['matrix'][0]),
                             output_length=len(self.phrase[0]['matrix'][0]),


                              )
        #self.model.add(seq)
        #self.model.add(Activation('softmax',
        #                          name="activation_1"))
        # self.model.add(TimeDistributed(Dense(1, activation='softmax')))

        # Compile with sgd
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
        self.model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
        #self.model.compile(loss='mse', optimizer='rmsprop')
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

        for line in predictions[0]:
            print('[' + ' | '.join(map(str, line)) + ']')

        self.log.info('Finished')
