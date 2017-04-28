# -*- coding: utf-8 -*-

import os

from rorschach.prediction.common import BasePredictor, CallbackRunner, KerasCallbackRunnerBridge
from rorschach.utilities import Config, LoggerWrapper


class BaseKerasPredictor(BasePredictor):

    def __init__(self):
        super().__init__()

        self.log = LoggerWrapper.load(__name__)
        self.model = None
        self.callback = None

        self.transformation_handlers = [
            # Initialize the labels
            'transformation.handlers.initializers.LabelInitializeHandler',

            # Noise
            'transformation.handlers.input.NoiseHandler',

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

    def build(self):
        self.callback = KerasCallbackRunnerBridge(
            CallbackRunner(self.data_container)
        )

    def compile(self):
        raise NotImplemented('Implement the compile function')

    def train(self):
        self.log.info('Begin')

        self.compile()

        self.model.fit(
            self.training_images_transformed,
            self.training_labels_transformed,
            epochs=Config.get('predicting.epochs'),
            verbose=1,
            batch_size=Config.get('predicting.batch-size'),
            validation_data=(self.validate_images_transformed, self.validate_labels_transformed),
            callbacks=[self.callback]
        )

        self.log.info('Finish')

    def load(self):
        weight_file = Config.get_path('path.output', 'model.h5', fragment=Config.get('uid'))
        if not os.path.exists(weight_file):
            raise Exception('Weights file not found in ', weight_file)

        self.model.load_weights(weight_file)

    def test(self):
        self.log.info('Begin test')

        self.load()
        self.compile()

        loss_and_metrics = self.model.evaluate(
            self.test_images_transformed,
            self.test_labels_transformed,
            batch_size=Config.get('predicting.batch-size')
        )

        print(loss_and_metrics)

        self.log.info('Finish test')
