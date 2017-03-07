#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.optimizers import Adam
from keras.layers import LSTM, Activation, Dense, Dropout, Embedding, InputLayer, TimeDistributed
from keras.models import Sequential
from keras.regularizers import ActivityRegularizer, WeightRegularizer
from keras.utils.visualize_util import plot

from rorschach.prediction.common import BasePredictor, CallbackRunner, KerasCallbackRunnerBridge
from rorschach.prediction.keras.tools import DimCalculator
from rorschach.utilities import Config, LoggerWrapper


class LSTMEmbeddingPredictor(BasePredictor):

    def __init__(self):
        super().__init__()

        self.log = LoggerWrapper.load(__name__)
        self.model = None
        self.callback = None

        self.transformation_handlers = [
            # Initialize the labels
            'transformation.handlers.initializers.LabelInitializeHandler',

            # Translate individual bits to string representations
            'transformation.handlers.input.ConcatenateBinaryDataHandler',

            # Pad the input sequence to fit the longest sequence
            'transformation.handlers.input.PadHandler',

            # Translate text sequences into integers (1B -> -1, 6W -> 6, ...)
            'transformation.handlers.input.IntegerifyStringSequenceSpatialHandler',

            # Rearrange values from previous handler so the are all positive integers starting from 0
            'transformation.handlers.input.RearrangeSequenceValuesHandler',

            # Translate the label text to corresponding integer ids (A -> 1, D -> 4, ...)
            'transformation.handlers.output.IntegerifyLabelHandler',

            # Keras specific handler. Change the output length to the same as the input (LSTMs)
            'transformation.handlers.output.KerasHandler',

            # Swap inputs and labels
            'transformation.handlers.finalize.SwapHandler'
        ]

    def prepare(self):
        input_width = DimCalculator.width(self.training_images_transformed)
        output_depth = DimCalculator.depth()

        self.callback = KerasCallbackRunnerBridge(
            CallbackRunner(self.data_container)
        )

        self.model = Sequential()
        self.model.add(InputLayer(batch_input_shape=(Config.get('predicting.batch_size'), input_width)))
        self.model.add(Embedding(1024, output_depth, mask_zero=True))
        self.model.add(LSTM(output_dim=256,
                            return_sequences=True,
                            W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                            b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
                            ))

        self.model.add(Dropout(0.2))

        self.model.add(TimeDistributed(Dense(output_dim=output_depth)))

        self.model.add(Activation('softmax'))

        optimizer = Adam()

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=[
                'categorical_accuracy'
            ]
        )

        self.model.summary()

        # Bootstrap our model to the bridge
        self.callback.model = self.model

        plot(self.model, to_file='model_rnn.png', show_shapes=True)

    def train(self):
        self.log.info('Begin')

        self.model.fit(self.test_images_transformed,
                       self.test_labels_transformed,
                       nb_epoch=Config.get('predicting.epochs'),
                       verbose=1,
                       batch_size=Config.get('predicting.batch-size'),
                       validation_data=(self.validate_images_transformed, self.validate_labels_transformed),
                       callbacks=[self.callback]
                       )

        self.log.info('Finish')

    def test(self):
        self.log.info('Begin test')

        self.log.info('Finish test')
