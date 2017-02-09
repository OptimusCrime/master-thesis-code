#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler


class RearrangeSequenceValuesHandler(BaseHandler):

    def __init__(self):
        super().__init__()

        self.bounds = {
            'upper': None,
            'lower': None
        }
        self.rearrange = False

    def run(self, input_lists):
        super().run(input_lists)

        self.rearrange = True

        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        # Ignore the data set
        if key == DataSetTypes.LETTER_SET:
            return

        super().list_handler(input_list, key)

        return input_list

    def obj_handler(self, obj):
        if self.rearrange:
            return self.rearrange_values(obj)

        self.find_bounds(obj)

        return obj

    def rearrange_values(self, obj):
        input_sequence = obj[DataSetTypes.IMAGES]['input']
        new_matrix = np.zeros(input_sequence.shape, dtype=np.int64)
        for i in range(len(input_sequence)):
            new_matrix[i] = input_sequence[i] + abs(self.bounds['lower'])

        # Swap array
        obj[DataSetTypes.IMAGES]['input_unrearranged'] = obj[DataSetTypes.IMAGES]['input']
        obj[DataSetTypes.IMAGES]['input'] = new_matrix

        return obj

    def find_bounds(self, obj):
        for val in obj[DataSetTypes.IMAGES]['input']:
            if self.bounds['lower'] is None or val < self.bounds['lower']:
                self.bounds['lower'] = val
            if self.bounds['upper'] is None or val > self.bounds['upper']:
                self.bounds['upper'] = val
