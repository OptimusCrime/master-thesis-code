#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utilities import Config, Filesystem, unpickle_data


class PredictorWrapper:

    def __init__(self):
        self.predictor = None

    def run(self):
        assert(self.predictor is not None)

        if Config.get('preprocessing.mode') == 'normal':
            self.predictor.data_set = unpickle_data(Filesystem.get_root_path('data/data_set.pickl'))
        else:
            self.predictor.data_set = unpickle_data(Filesystem.get_root_path('data/word_set.pickl'))

        self.predictor.phrase = unpickle_data(Filesystem.get_root_path('data/phrase.pickl'))

        self.predictor.preprocess()
        self.predictor.train()
        self.predictor.predict()

    @property
    def predictions(self):
        return self.predictor.predictions
