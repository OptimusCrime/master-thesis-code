# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler
from rorschach.utilities import Config


class SwapHandler(BaseHandler):

    def __init__(self):
        super().__init__()

    def run(self, input_lists):
        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        if key == DataSetTypes.LETTER_SET:
            return

        inputs = SwapHandler.reshape_input(input_list)
        labels = SwapHandler.reshape_labels(input_list)

        new_list = {
            DataSetTypes.IMAGES: inputs,
            DataSetTypes.LABELS: labels
        }

        self.data['voc_size_labels'] = len(Config.get('general.characters')) + 1
        self.data['voc_size_input'] = SwapHandler.calculate_voc_size(inputs)

        return new_list

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

    @staticmethod
    def calculate_voc_size(input_list):
        voc_size = 0
        for seq in input_list:
            for val in seq:
                if val > voc_size:
                    voc_size = val

        return voc_size + 1
