#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorchach.transformation import Transformator
from rorchach.utilities import Config, Filesystem, unpickle_data


class PredictorWrapper:

    def __init__(self):
        self.predictor = None

    def run(self):
        assert(self.predictor is not None)

        self.predictor.data_set = unpickle_data(Filesystem.get_root_path('data/word_set.pickl'))
        self.predictor.phrase = unpickle_data(Filesystem.get_root_path('data/phrase.pickl'))

        if Config.get('transformation.run'):
            self.transform()

        self.predictor.preprocess()
        self.predictor.train()
        self.predictor.predict()

    def transform(self):
        transformator = Transformator()
        transformator.data_lists = [self.predictor.data_set, self.predictor.phrase]
        transformator.run()

    @property
    def predictions(self):
        return self.predictor.predictions
