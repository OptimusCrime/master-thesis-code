# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler


class Reshape3DHandler(BaseHandler):

    def __init__(self):
        super().__init__()

    def run(self, input_lists):
        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        if key == DataSetTypes.DATA_SET:
            return

        input_list[DataSetTypes.IMAGES] = self.reshape_input(input_list[DataSetTypes.IMAGES])
        input_list[DataSetTypes.LABELS] = self.reshape_labels(input_list[DataSetTypes.LABELS])

    def reshape_input(self, input_list):
        raw_array = []
        for obj in input_list:
            raw_array.append(obj['input'])

        np_array = np.array(raw_array)
        return np_array.reshape((np_array.shape[0], np_array.shape[1], 1))

    def reshape_labels(self, labels_list):
        raw_array = []
        for obj in labels_list:
            raw_array.append(obj['value'])

        return np.array(raw_array)
