#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import NaiveNNPredictor
from utilities import Filesystem, unpickle_data


class PredictorWrapper:

    def __init__(self):
        self.predictor = NaiveNNPredictor()

    def run(self):
        self.predictor.data_set = unpickle_data(Filesystem.get_root_path('data/data_set.pickl'))
        self.predictor.phrase = unpickle_data(Filesystem.get_root_path('data/phrase.pickl'))

        self.predictor.preprocess()
        self.predictor.predict()

    @property
    def predictions(self):
        return self.predictor.predictions
