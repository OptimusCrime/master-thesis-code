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

        self.predictor.training_set = unpickle_data(Filesystem.get_root_path('data/training_set.pickl'))
        self.predictor.test_set = unpickle_data(Filesystem.get_root_path('data/test_set.pickl'))
        self.predictor.verification_set = unpickle_data(Filesystem.get_root_path('data/verification_set.pickl'))

        if Config.get('transformation.run'):
            self.transform()

        self.predictor.prepare()
        self.predictor.train()
        self.predictor.predict()

    def transform(self):
        transformator = Transformator()

        transformator.construct_lists([{
            'set': unpickle_data(Filesystem.get_root_path('data/letter_set.pickl')),
            'type': DataSetTypes.LETTER_SET
        }, {
            'set': self.predictor.training_set,
            'type': DataSetTypes.TRAINING_SET
        }, {
            'set': self.predictor.test_set,
            'type': DataSetTypes.TEST_SET
        }, {
            'set': self.predictor.verification_set,
            'type': DataSetTypes.VERIFICATION_SET
        }])

        transformator.run()

        self.predictor.training_images_transformed, \
            self.predictor.training_labels_transformed = transformator.data_set(DataSetTypes.TRAINING_SET)

        self.predictor.test_images_transformed, \
            self.predictor.test_labels_transformed = transformator.data_set(DataSetTypes.TEST_SET)

        self.predictor.verification_images_transformed, \
        self.predictor.verification_labels_transformed = transformator.data_set(DataSetTypes.VERIFICATION_SET)

        self.predictor.data = transformator.data
