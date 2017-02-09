#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler


class IntegerifyStringSequenceSpatialHandler(BaseHandler):

    def __init__(self):
        super().__init__()

    def run(self, input_lists):
        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        # Ignore the data set
        if key == DataSetTypes.LETTER_SET:
            return

        super().list_handler(input_list, key)

        return input_list

    def obj_handler(self, obj):
        input_sequence = obj[DataSetTypes.IMAGES]['input']
        new_matrix = np.zeros(input_sequence.shape, dtype=np.int64)
        for i in range(len(input_sequence)):
            if input_sequence[i] == '0':
                continue

            # We use negative values for black pixels and positive values for white pixels. We do this by multiplying
            # our value with the constant defined here
            val_type = input_sequence[i][-1]
            val_const = 1
            if val_type == 'B':
                val_const = -1

            # Take the original value (11W), remove the last char (11) multiply it with our constant
            new_matrix[i] = int(input_sequence[i][:-1]) * val_const

        # Swap array
        obj[DataSetTypes.IMAGES]['input_str'] = obj[DataSetTypes.IMAGES]['input']
        obj[DataSetTypes.IMAGES]['input'] = new_matrix

        return obj
