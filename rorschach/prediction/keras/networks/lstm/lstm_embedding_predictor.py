#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import LSTM, Activation, Dense, Dropout, Embedding, InputLayer
from keras.models import Sequential
from keras.regularizers import ActivityRegularizer, WeightRegularizer
from keras.utils.visualize_util import plot

from rorschach.prediction.common import BasePredictor
from rorschach.prediction.keras.callbacks.plotter import PlotCallback
from rorschach.prediction.keras.tools import DimCalculator
from rorschach.utilities import Config, LoggerWrapper  # isort:skip


class LSTMEmbeddingPredictor(BasePredictor):

    def __init__(self):
        super().__init__()

        self.log = LoggerWrapper.load(__name__)
        self.model = None
        self.callback = None

    def prepare(self):
        input_width = DimCalculator.width(self.training_images_transformed)
        output_depth = DimCalculator.depth()

        self.callback = PlotCallback()
        self.callback.epochs = Config.get('predicting.epochs')

        self.model = Sequential()
        self.model.add(InputLayer(batch_input_shape=(Config.get('predicting.batch_size'), input_width)))
        self.model.add(Embedding(1024, output_depth))
        self.model.add(LSTM(output_dim=256,
                            return_sequences=True,
                            W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                            b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
                            ))

        self.model.add(Dropout(0.2))

        self.model.add(Dense(output_dim=output_depth))

        self.model.add(Activation('softmax'))

        self.model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

        self.model.summary()

        plot(self.model, to_file='model_rnn.png', show_shapes=True)

    def train(self):
        self.log.info('Begin')

        self.model.fit(self.training_images_transformed,
                       self.training_labels_transformed,
                       nb_epoch=Config.get('predicting.epochs'),
                       verbose=1,
                       batch_size=Config.get('predicting.batch_size'),
                       validation_data=(self.validate_images_transformed, self.validate_labels_transformed),
                       callbacks=[self.callback]
                       )

        self.log.info('Finish')

    def predict(self):
        self.log.info('Begin predicting')
