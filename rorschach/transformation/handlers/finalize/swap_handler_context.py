# -*- coding: utf-8 -*-

import numpy as np

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler
from rorschach.utilities import Config, unpickle_data


'''
SwapHandler

This handler does swapping of images and labels. It also calculates the widths for various things.

'''


class SwapContextHandler(BaseHandler):

    def __init__(self):
        super().__init__()

        self.voc_sizes = False

        self.input_vocabulary_upper = 0

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

        inputs = SwapContextHandler.reshape_input(input_list)
        labels = SwapContextHandler.reshape_labels(input_list)

        new_list = {
            DataSetTypes.IMAGES: inputs,
            DataSetTypes.LABELS: labels
        }

        return new_list

    def calculate_voc_size(self):
        self.voc_sizes = True

        Config.set('dataset.voc_size_input', self.input_vocabulary_upper)
        Config.set('dataset.voc_size_output', len(unpickle_data(Config.get_path('path.data', 'labels.pickl'))))

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
        self.input_vocabulary_upper = (len(input_list[0][DataSetTypes.IMAGES]['input']) * 2) + 2
