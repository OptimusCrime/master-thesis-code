#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.common import DataSetTypes
from rorschach.transformation import Transformator
from rorschach.utilities import Config, Filesystem, unpickle_data


class PredictorWrapper:

    def __init__(self):
        self.predictor = None

    def run(self):
        assert(self.predictor is not None)

        self.predictor.data_set = unpickle_data(Filesystem.get_root_path('data/data_set.pickl'))
        self.predictor.word_set = unpickle_data(Filesystem.get_root_path('data/word_set.pickl'))
        self.predictor.phrase = unpickle_data(Filesystem.get_root_path('data/phrase.pickl'))

        if Config.get('transformation.run'):
            self.transform()

        self.predictor.preprocess()
        self.predictor.train()
        self.predictor.predict()

    def transform(self):
        transformator = Transformator()

        transformator.construct_lists([{
            'set': self.predictor.data_set,
            'type': DataSetTypes.DATA_SET
        }, {
            'set': self.predictor.word_set,
            'type': DataSetTypes.WORD_SET
        }, {
            'set': self.predictor.phrase,
            'type': DataSetTypes.PHRASE
        }])

        transformator.run()

        self.predictor.training_images_transformed, \
            self.predictor.training_labels_transformed = transformator.data_set(DataSetTypes.WORD_SET)

        self.predictor.predicting_image_transformed, \
            self.predictor.predicting_label_transformed = transformator.data_set(DataSetTypes.PHRASE)

    @property
    def predictions(self):
        return self.predictor.predictions
