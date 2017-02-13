# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler


class ReshapeBestHandler(BaseHandler):

    def __init__(self):
        super().__init__()

    def run(self, input_lists):
        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        if key == DataSetTypes.LETTER_SET:
            return

        new_list = {DataSetTypes.IMAGES: self.reshape_input(input_list),
                    DataSetTypes.LABELS: self.reshape_labels(input_list)}

        return new_list

    def reshape_input(self, input_list):
        raw_array = []
        for obj in input_list:
            raw_array.append(obj[DataSetTypes.IMAGES]['input'])

        np_array = np.array(raw_array)
        return np_array

    def reshape_labels(self, labels_list):
        raw_array = [[] for _ in range(10)]

        for obj in labels_list:
            for i in range(10):
                raw_array[i].append(obj[DataSetTypes.LABELS]['value'][i])

        for i in range(10):
            raw_array[i] = np.array(raw_array[i])

        raw_array.reverse()

        return raw_array
