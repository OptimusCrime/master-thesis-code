# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler


class ReshapeCorrectHandler(BaseHandler):

    def __init__(self):
        super().__init__()

    def run(self, input_lists):
        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        if key == DataSetTypes.LETTER_SET:
            return

        input_list[DataSetTypes.IMAGES] = self.reshape_input(input_list[DataSetTypes.IMAGES])
        input_list[DataSetTypes.LABELS] = self.reshape_labels(input_list[DataSetTypes.LABELS])

    def reshape_input(self, input_list):
        raw_array = []
        for obj in input_list:
            raw_array.append(self.reshape_actual_input(obj['input']))

        np_array = np.array(raw_array)
        return np_array

    def reshape_actual_input(self, ipt):
        return ipt

    def reshape_labels(self, labels_list):
        raw_array = []
        for obj in labels_list:
            raw_array.append(obj['value'])

        return np.array(raw_array)