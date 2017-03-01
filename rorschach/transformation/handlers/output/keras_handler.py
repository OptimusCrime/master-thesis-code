#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler


class KerasHandler(BaseHandler):

    def __init__(self):
        super().__init__()

        self.length = None

    def run(self, input_lists):
        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        # Ignore the data set
        if key == DataSetTypes.LETTER_SET:
            return

        if self.length is None:
            self.calculate_length(input_list)

        super().list_handler(input_list, key)

        return input_list

    def calculate_length(self, input_list):
        if len(input_list) > 0:
            self.length = len(input_list[0][DataSetTypes.IMAGES]['input'])

    def obj_handler(self, obj):
        label_matrix = np.zeros(self.length, dtype=np.int32)
        labels = obj[DataSetTypes.LABELS]['value']
        for i in range(len(labels)):
            if labels[i] == 0:
                break

            label_matrix[i] = labels[i]

        # Swap old and new
        obj[DataSetTypes.LABELS]['value_short'] = labels
        obj[DataSetTypes.LABELS]['value'] = label_matrix

        return obj
