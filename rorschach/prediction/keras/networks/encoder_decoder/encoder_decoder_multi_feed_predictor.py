#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import (Activation, Dense, Dropout, TimeDistributed, Embedding, Input)
from keras.models import Model
from keras.optimizers import SGD
from keras.utils.visualize_util import plot

from rorschach.prediction.keras.callbacks import CallbackWrapper
from rorschach.prediction.keras.callbacks.plotter import PlotCallback
from rorschach.prediction.keras.layers import HiddenStateLSTM2
from rorschach.prediction.common import BasePredictor
from rorschach.utilities import Config, LoggerWrapper  # isort:skip


class EncoderDecoderMultiFeedPredictor(BasePredictor):

    def __init__(self):
        super().__init__()

        self.log = LoggerWrapper.load(__name__)
        self.model = None
        self.callback = None

    def prepare(self):
        self.keras_setup()

    def keras_setup(self):
        self.callback = CallbackWrapper([PlotCallback])
        self.callback.epochs = Config.get('predicting.epochs')

        enc_input = Input(shape=(48,), dtype='int32', name='encoder_input')
        enc_layer = Embedding(300, 19, mask_zero=True)(enc_input)
        enc_layer, *hidden = HiddenStateLSTM2(1024, dropout_W=0.5, dropout_U=0.5, return_sequences=False)(enc_layer)

        dec_input = Input(shape=(48,), dtype='int32', name='decoder_input')
        dec_layer = Embedding(300, 19, mask_zero=True)(dec_input)
        dec_layer, _, _ = HiddenStateLSTM2(1024, dropout_W=0.5, dropout_U=0.5, return_sequences=True)([dec_layer] + hidden)
        dec_layer = TimeDistributed(Dense(19))(dec_layer)

        dec_output = Dropout(0.2)(dec_layer)
        dec_output = Activation('softmax', name='decoder_output')(dec_output)

        self.model = Model(input=[enc_input, dec_input], output=dec_output)

        sgd = SGD(lr=1., decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.summary()

        plot(self.model, to_file='model_rnn.png', show_shapes=True)

    def train(self):
        self.log.info('Begin training')
        self.model.fit([self.training_images_transformed, self.training_images_transformed],
                       self.training_labels_transformed,
                       nb_epoch=Config.get('predicting.epochs'),
                       verbose=1,
                       batch_size=Config.get('predicting.batch_size'),
                       validation_data=([self.test_images_transformed, self.test_images_transformed], self.test_labels_transformed),
                       callbacks=[self.callback]
                       )

        self.log.info('Finished training')

    def predict(self):
        self.log.info('Begin predicting')

        derp = self.model.predict(self.test_images_transformed, batch_size=100)
        print(list(derp[0:2]))

