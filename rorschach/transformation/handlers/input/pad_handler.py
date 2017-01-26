#!/usr/bin/env python
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

    def run(self, input_lists):
        super().run(input_lists)

        self.pad = True

        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        # Ignore the data set
        if key == DataSetTypes.LETTER_SET:
            return

        # Require the sequence key
        if not self.pad and len(input_list[DataSetTypes.IMAGES]) > 0:
            assert 'sequence' in input_list[DataSetTypes.IMAGES][0]

        super().list_handler(input_list, key)

    def obj_handler(self, ipt, label):
        if self.pad:
            return self.pad_input(ipt)

        width_sequence = len(ipt['sequence'])
        if self.widest_sequence is None or width_sequence > self.widest_sequence:
            self.widest_sequence = width_sequence

    def pad_input(self, obj):
        new_matrix = np.full(self.widest_sequence, 0, dtype=(np.str, 35))
        for v in range(len(obj['sequence'])):
            new_matrix[v] = obj['sequence'][v]

        obj['input'] = new_matrix
