#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.prediction.common import BasePredictor
from rorschach.prediction.tensorflow.callbacks import CallbackRunner
from rorschach.prediction.tensorflow.callbacks.plotter import CallbackPlotter
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

        self.model.callback = CallbackRunner([
            CallbackPlotter
        ])

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
