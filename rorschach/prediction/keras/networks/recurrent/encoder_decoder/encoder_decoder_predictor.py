#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import (Activation, Dense, Dropout, TimeDistributed, Embedding, Input)
from keras.models import Model
from keras.optimizers import SGD
from keras.utils.visualize_util import plot
from rorschach.prediction.callbacks import CallbackWrapper

from prediction.keras.callbacks.plotter import PlotCallback
from prediction.keras.layers import HiddenStateLSTM
from prediction.keras.networks import BasePredictor
from rorschach.utilities import Config, LoggerWrapper  # isort:skip


class EncodeDecodePredictor(BasePredictor):

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
        enc_layer, *hidden = HiddenStateLSTM(1024, dropout_W=0.5, dropout_U=0.5, return_sequences=False)(enc_layer)

        dec_input = Input(shape=(48,), dtype='int32', name='decoder_input')
        dec_layer = Embedding(300, 19, mask_zero=True)(dec_input)
        dec_layer, _, _ = HiddenStateLSTM(1024, dropout_W=0.5, dropout_U=0.5, return_sequences=True)([dec_layer] + hidden)
        dec_layer = TimeDistributed(Dense(19))(dec_layer)

        dec_output = Dropout(0.2)(dec_layer)
        dec_output = Activation('softmax', name='decoder_output')(dec_output)

        self.model = Model(input=[enc_input, dec_input], output=dec_output)

        sgd = SGD(lr=1., decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        '''self.model = Sequential()
        self.model.add(InputLayer(batch_input_shape=(Config.get('predicting.batch_size'), 48)))
        self.model.add(Embedding(300, 19))
        self.model.add(LSTM(output_dim=256,
                                          return_sequences=True,
                                          W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                                          b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
                                          stateful=False,

                                          ),
                                     )

        self.model.add(Dropout(0.2))

        self.model.add(Dense(output_dim=19))

        self.model.add(Activation('softmax'))

        self.model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
        '''
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

