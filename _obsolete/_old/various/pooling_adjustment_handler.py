# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from prediction.common.helpers import PoolingFactorCalculator

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler
from rorschach.utilities import Filesystem, pickle_data


class PoolingAdjustmentHandler(BaseHandler):

    def __init__(self):
        super().__init__()

        self.calculated = False
        self.widest_sequence = None
        self.widest_label = None

        self.adjust_sequence = None
        self.adjust_label = None

    def run(self, input_lists):
        super().run(input_lists)

        self.calculate_widths()

        if self.adjust_sequence is not None or self.adjust_label is not None:
            super().run(input_lists)

        return input_lists

    def calculate_widths(self):
        self.calculated = True

        pooling = PoolingFactorCalculator.calc(self.widest_sequence, self.widest_label, PoolingFactorCalculator.ADJUST)

        pickle_data(pooling, Filesystem.get_root_path('data/pooling.pickl'))

        if pooling['width_sequence'] != self.widest_sequence:
            self.adjust_sequence = pooling['width_sequence']

        if pooling['width_label'] != self.widest_label:
            self.adjust_label = pooling['width_label']

    def list_handler(self, input_list, key):
        # Ignore the data set
        if key == DataSetTypes.LETTER_SET:
            return

        super().list_handler(input_list, key)

        return input_list

    def obj_handler(self, obj):
        if not self.calculated:
            return self.evaluate_lengths(obj)

        if self.adjust_sequence is not None:
            obj = self.do_adjust_sequence(obj)

        # if self.adjust_label is not None:
        #    self.do_adjust_label(obj)

        return obj

    def evaluate_lengths(self, obj):
        width_sequence = len(obj[DataSetTypes.IMAGES]['input'])
        if self.widest_sequence is None or width_sequence > self.widest_sequence:
            self.widest_sequence = width_sequence

        width_label = len(obj[DataSetTypes.IMAGES]['text'])
        if self.widest_label is None or width_label > self.widest_label:
            self.widest_label = width_label

        return obj

    def do_adjust_sequence(self, obj):
        new_matrix = np.full(self.adjust_sequence, 0, dtype=np.int)
        for v in range(len(obj[DataSetTypes.IMAGES]['input'])):
            new_matrix[v] = obj[DataSetTypes.IMAGES]['input'][v]

        obj[DataSetTypes.IMAGES]['input'] = new_matrix

        return obj

    def do_adjust_label(self, obj):
        # Do we need to implement this?
        pass
