# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler


class ReshapeGoodHandler(BaseHandler):

    def __init__(self):
        super().__init__()

    def run(self, input_lists):
        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        if key == DataSetTypes.LETTER_SET:
            return

        new_list = {
            DataSetTypes.IMAGES: None,
            DataSetTypes.LABELS: None
        }

        new_list[DataSetTypes.IMAGES] = self.reshape_input(input_list)
        new_list[DataSetTypes.LABELS] = self.reshape_labels(input_list)

        return new_list

    def reshape_input(self, input_list):
        raw_array = [[] for _ in range(5)]
        for obj in input_list:
            reshaped = self.reshape_actual_input(obj[DataSetTypes.IMAGES]['input'])
            for i in range(5):
                raw_array[i].append(reshaped[i])

        for i in range(5):
            raw_array[i] = np.array(raw_array[i])

        return raw_array

    def reshape_actual_input(self, ipt):
        new_matrix = np.full((5, 10), 0, dtype=np.int)
        bucket = 0
        offset = 0
        for i in range(len(ipt)):
            new_matrix[bucket][offset] = ipt[i]
            offset += 1
            if offset == 10:
                bucket += 1
                offset = 0
        return new_matrix

    def reshape_labels(self, labels_list):
        raw_array = [[] for _ in range(10)]

        for obj in labels_list:
            for i in range(10):
                raw_array[i].append(obj[DataSetTypes.LABELS]['value'][i])

        for i in range(10):
            raw_array[i] = np.array(raw_array[i])

        raw_array.reverse()

        return raw_array
