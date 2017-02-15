# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler
from rorschach.utilities import Config, unpickle_data


class SwapHandler(BaseHandler):

    def __init__(self):
        super().__init__()

        self.voc_sizes = False

    def prepare(self):
        super().prepare()

        self.data['voc_lower'] = 0
        self.data['voc_upper'] = 0

    def run(self, input_lists):
        super().run(input_lists)

        self.calculate_voc_size()

        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        if key == DataSetTypes.LETTER_SET:
            return

        if not self.voc_sizes:
            self.count_voc_size(input_list)
            return input_list

        inputs = SwapHandler.reshape_input(input_list)
        labels = SwapHandler.reshape_labels(input_list)

        new_list = {
            DataSetTypes.IMAGES: inputs,
            DataSetTypes.LABELS: labels
        }

        return new_list

    def calculate_voc_size(self):
        self.voc_sizes = True

        self.data['voc_size_labels'] = len(unpickle_data(Config.get_path('path.data', 'labels.pickl')))
        self.data['voc_size_input'] = self.data['voc_lower'] + self.data['voc_upper']

    @staticmethod
    def reshape_input(input_list):
        raw_array = []
        for obj in input_list:
            raw_array.append(obj[DataSetTypes.IMAGES]['input'])

        return np.array(raw_array)

    @staticmethod
    def reshape_labels(labels_list):
        raw_array = []
        for obj in labels_list:
            raw_array.append(obj[DataSetTypes.LABELS]['value'])

        return np.array(raw_array)

    def count_voc_size(self, input_list):
        for seq in input_list:
            for val in seq[DataSetTypes.IMAGES]['input']:
                if val < self.data['voc_lower']:
                    self.data['voc_lower'] = val

                if val > self.data['voc_upper']:
                    self.data['voc_upper'] = val
