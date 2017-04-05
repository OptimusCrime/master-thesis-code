# -*- coding: utf-8 -*-

import numpy as np

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler


class PadHandler(BaseHandler):

    def __init__(self):
        super().__init__()

        self.pad = False
        self.widest_sequence = None
        self.widest_label = None
        self.sequence_key = None

    def run(self, input_lists):
        super().run(input_lists)

        self.pad = True

        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        # Ignore the data set
        if key == DataSetTypes.LETTER_SET:
            return

        if self.sequence_key is None:
            if 'sequence' in input_list[0][DataSetTypes.IMAGES]:
                self.sequence_key = 'sequence'
            else:
                self.sequence_key = 'concatenated_binary'

        super().list_handler(input_list, key)

        return input_list

    def obj_handler(self, obj):
        if self.pad:
            return self.pad_input(obj)

        width_sequence = len(obj[DataSetTypes.IMAGES][self.sequence_key])
        if self.widest_sequence is None or width_sequence > self.widest_sequence:
            self.widest_sequence = width_sequence

        return obj

    def pad_input(self, obj):
        new_matrix = np.full(self.widest_sequence, 0, dtype=(np.str, 35))
        for v in range(len(obj[DataSetTypes.IMAGES][self.sequence_key])):
            new_matrix[v] = obj[DataSetTypes.IMAGES][self.sequence_key][v]

        obj[DataSetTypes.IMAGES]['input'] = new_matrix

        return obj
