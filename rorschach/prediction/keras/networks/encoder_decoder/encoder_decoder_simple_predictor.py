#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import (Activation, Dense, TimeDistributed, Input, RepeatVector)
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import WeightRegularizer, ActivityRegularizer
from keras.utils.visualize_util import plot

from rorschach.prediction.common import BasePredictor
from rorschach.prediction.keras.callbacks import CallbackWrapper
from rorschach.prediction.keras.callbacks.plotter import PlotCallback
from rorschach.prediction.keras.layers import HiddenStateLSTM2
from rorschach.utilities import Config, LoggerWrapper  # isort:skip


class EncoderDecoderSimplePredictor(BasePredictor):

    def __init__(self):
        super().__init__()

        self.log = LoggerWrapper.load(__name__)
        self.model = None
        self.callback = None

    def prepare(self):
        self.callback_setup()
        self.keras_setup()

    def callback_setup(self):
        self.callback = CallbackWrapper([PlotCallback])
        self.callback.data['epochs'] = Config.get('predicting.epochs')

    def keras_setup(self):
        ipt = Input(shape=(48, self.training_images_transformed.shape[-1]))

        encoder, *hidden = HiddenStateLSTM2(
            1024,
            dropout_W=0.2,
            dropout_U=0.2,
            W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
            b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
            return_sequences=False
        )(ipt)

        repeator = RepeatVector(10)(encoder)

        decoder, _, _ = HiddenStateLSTM2(
            1024,
            dropout_W=0.2,
            dropout_U=0.2,
            W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
            b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
            return_sequences=True
        )([repeator, hidden[0], hidden[1]])

        output = TimeDistributed(Dense(19))(decoder)
        output = Activation('softmax')(output)

        self.model = Model(input=ipt, output=output)

        sgd = SGD(lr=1., decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.summary()

        plot(self.model, to_file='test1.png', show_shapes=True)

    def train(self):
        self.log.info('Begin training')

        self.model.fit(
            self.training_images_transformed,
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

        derp = self.model.predict(
            self.test_images_transformed,
            batch_size=100
        )

        print(list(derp[0:2]))
