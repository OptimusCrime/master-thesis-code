#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import (Activation)
from keras.models import Sequential
from keras.utils.visualize_util import plot
from seq2seq.models import Seq2Seq

from rorschach.prediction.common import BasePredictor
from rorschach.prediction.keras.callbacks.plotter import PlotCallback
from rorschach.utilities import Config, LoggerWrapper  # isort:skip


class EmbeddingRNNSeq2SeqPredictor(BasePredictor):

    def __init__(self):
        super().__init__()

        self.log = LoggerWrapper.load(__name__)
        self.model = None
        self.callback = None

    def prepare(self):
        pass

    def train(self):
        self.log.info('Begin training')
        pass

        self.log.info('Finished training')

    def predict(self):
        self.log.info('Begin predicting')
        pass
