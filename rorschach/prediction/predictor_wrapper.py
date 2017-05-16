# -*- coding: utf-8 -*-

import json
import os

from rorschach.common import DataSetTypes
from rorschach.transformation import Transformator
from rorschach.utilities import Config, JsonConfigEncoder, unpickle_data


class PredictorWrapper:

    def __init__(self):
        self.predictor = None

    def setup(self):
        assert (self.predictor is not None)

        self.predictor.training_set = unpickle_data(Config.get_path('path.data', 'training_set.pickl'))
        self.predictor.validate_set = unpickle_data(Config.get_path('path.data', 'validate_set.pickl'))
        self.predictor.test_set = unpickle_data(Config.get_path('path.data', 'test_set.pickl'))

        if len(self.predictor.transformation_handlers) > 0:
            self.transform()
            self.dump_info()

        self.predictor.build()

    def train(self):
        self.setup()

        if Config.get('general.mode') == 'continue':
            self.predictor.load()

        self.predictor.train()

    def test(self):
        self.setup()

        if Config.get('general.mode') == 'predict':
            return self.predictor.predict()
        self.predictor.test()

    def transform(self):
        transformator = Transformator(self.predictor.transformation_handlers)

        transformator.construct_lists([{
            'set': unpickle_data(Config.get_path('path.data', 'letter_set.pickl')),
            'type': DataSetTypes.LETTER_SET
        }, {
            'set': self.predictor.training_set,
            'type': DataSetTypes.TRAINING_SET
        }, {
            'set': self.predictor.validate_set,
            'type': DataSetTypes.VALIDATE_SET
        }, {
            'set': self.predictor.test_set,
            'type': DataSetTypes.TEST_SET
        }])

        transformator.run()

        self.predictor.training_images_transformed, \
            self.predictor.training_labels_transformed = transformator.data_set(DataSetTypes.TRAINING_SET)

        self.predictor.validate_images_transformed, \
            self.predictor.validate_labels_transformed = transformator.data_set(DataSetTypes.VALIDATE_SET)

        self.predictor.test_images_transformed, \
            self.predictor.test_labels_transformed = transformator.data_set(DataSetTypes.TEST_SET)

    def dump_info(self):
        config_file = Config.get_path('path.output', 'config.json', fragment=Config.get('uid'))
        if not os.path.exists(config_file):
            self.dump_config()

        dataset_file = Config.get_path('path.output', 'dataset.json', fragment=Config.get('uid'))
        if not os.path.exists(dataset_file):
            self.dump_dataset()

    def dump_config(self):
        with open(Config.get_path('path.output', 'config.json', fragment=Config.get('uid')), 'w') as outfile:
            json.dump(Config.all(), outfile, cls=JsonConfigEncoder)

    def dump_dataset(self):
        data = {
            'training_size': len(self.predictor.training_images_transformed),
            'validate_size': len(self.predictor.validate_images_transformed),
            'test_size': len(self.predictor.test_images_transformed),
            'label_length_perhaps': len(self.predictor.training_labels_transformed[0])
        }

        with open(Config.get_path('path.output', 'dataset.json', fragment=Config.get('uid')), 'w') as outfile:
            json.dump(data, outfile)
