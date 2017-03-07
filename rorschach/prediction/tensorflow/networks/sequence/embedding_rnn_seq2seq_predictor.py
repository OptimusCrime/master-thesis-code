#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.prediction.common import BasePredictor
from rorschach.prediction.tensorflow.layers import Seq2Seq
from rorschach.prediction.tensorflow.tools import batch_gen

from rorschach.utilities import Config, LoggerWrapper  # isort:skip


class EmbeddingRNNSeq2SeqPredictor(BasePredictor):

    def __init__(self):
        super().__init__()

        self.log = LoggerWrapper.load(__name__)
        self.model = None
        self.callback = None

        self.training_batch_gen = None
        self.val_batch_gen = None

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

            # Swap inputs and labels
            'transformation.handlers.finalize.SwapHandler'
        ]

    def prepare(self):
        self.build_batches()
        self.build_model()

    def build_batches(self):
        self.train_batch_gen = batch_gen(
            self.training_images_transformed,
            self.training_labels_transformed,
            Config.get('predicting.batch-size')
        )

        self.validation_batch_gen = batch_gen(
            self.validate_images_transformed,
            self.validate_labels_transformed,
            Config.get('predicting.batch-size')
        )

        self.test_batch_gen = batch_gen(
            self.test_images_transformed,
            self.test_labels_transformed,
            Config.get('predicting.batch-size')
        )

    def build_model(self):
        self.log.info('Building model')

        self.model = Seq2Seq(
            xseq_len=self.training_images_transformed.shape[-1],
            yseq_len=self.training_labels_transformed.shape[-1],
            xvocab_size=self.data['voc_size_input'],
            yvocab_size=self.data['voc_size_labels'],
            emb_dim=1024,
            num_layers=3
        )

        self.model.training_set = self.train_batch_gen
        self.model.validation_set = self.validation_batch_gen
        self.model.test_set = self.test_batch_gen

        self.model.register_data_container(self.data_container)

        self.model.build_graph()

        self.log.info('Finished building model')

    def train(self):
        self.log.info('Begin')

        self.model.start_train()

        self.log.info('Finished')

    def test(self):
        self.log.info('Begin')

        self.model.start_test()

        self.log.info('Finished')
