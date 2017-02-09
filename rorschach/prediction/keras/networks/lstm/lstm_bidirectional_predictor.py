#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import LSTM, Activation, Bidirectional, Dense, Dropout, TimeDistributed
from keras.models import Sequential
from keras.regularizers import ActivityRegularizer, WeightRegularizer
from keras.utils.visualize_util import plot
from prediction.keras.callbacks.plotter import PlotCallback
from prediction.keras.networks import BasePredictor

from rorschach.utilities import Config, LoggerWrapper  # isort:skip


class LSTMBidirectionalPredictor(BasePredictor):

    def __init__(self):
        super().__init__()

        self.log = LoggerWrapper.load(__name__)
        self.model = None
        self.callback = None

    def prepare(self):
        self.keras_setup()

    def keras_setup(self):
        self.callback = PlotCallback()
        self.callback.epochs = Config.get('predicting.epochs')

        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(output_dim=256,
                                          return_sequences=True,
                                          W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                                          b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
                                          stateful=True),
                                     batch_input_shape=(Config.get('predicting.batch_size'), 30, 4)))

        self.model.add(Dropout(0.4))
        self.model.add(Bidirectional(LSTM(128,
                                          return_sequences=True,
                                          W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                                          b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
                                          stateful=True)))
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

        derp = self.model.predict(self.training_images_transformed, batch_size=Config.get('predicting.batch_size'))[0]
        print(list(derp[0:20]))
