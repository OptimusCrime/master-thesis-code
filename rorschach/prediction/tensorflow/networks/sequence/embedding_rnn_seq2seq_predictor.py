#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import (Activation)
from keras.models import Sequential
from keras.utils.visualize_util import plot
from seq2seq.models import Seq2Seq

from rorschach.prediction.common import BasePredictor
from rorschach.prediction.tensorflow.layers import Seq2Seq
from rorschach.prediction.tensorflow.tools import rand_batch_gen
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
        self.train_batch_gen = rand_batch_gen(
            self.training_images_transformed,
            self.training_labels_transformed,
            Config.get('predicting.batch_size')
        )

        self.val_batch_gen = rand_batch_gen(
            self.test_images_transformed,
            self.test_labels_transformed,
            Config.get('predicting.batch_size')
        )

    def build_model(self):
        self.log.info('Begin build model')

        self.model = Seq2Seq(
            xseq_len=self.training_images_transformed.shape[-1],
            yseq_len=self.training_labels_transformed.shape[-1],
            xvocab_size=self.data['voc_size_input'],
            yvocab_size=self.data['voc_size_labels'],
            ckpt_path='ckpt/twitter/', # <- TODO REMOVE
            emb_dim=1024,
            num_layers=3
        )

        self.log.info('Finished building model')

    def train(self):
        self.log.info('Begin training')

        # sess = self.model.restore_last_session()

        sess = self.model.train(self.train_batch_gen, self.val_batch_gen)

        self.log.info('Finished training')

    def predict(self):
        self.log.info('Begin predicting')
        pass
