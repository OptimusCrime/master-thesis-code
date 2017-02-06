#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler


class IntegerifyStringSequenceHandler(BaseHandler):

    def __init__(self):
        super().__init__()

        self.string_values = {}
        self.translation_lookup = None

    def run(self, input_lists):
        super().run(input_lists)

        self.create_translations()

        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        # Ignore the data set
        if key == DataSetTypes.LETTER_SET:
            return

        super().list_handler(input_list, key)

        return input_list

    def obj_handler(self, obj):
        if self.translation_lookup is not None:
            return self.apply_translation(obj)

        for val in obj[DataSetTypes.IMAGES]['input']:
            if val == '0':
                continue

            if val in self.string_values:
                self.string_values[val] += 1
            else:
                self.string_values[val] = 1

        return obj

    def create_translations(self):
        # We are now sorting the values descending by their popularity
        sorted_uniques_values = sorted(self.string_values, key=self.string_values.get, reverse=True)

        # Reassign the dict's values to be the index from the sorting, e.i. their popularity. The list still has a
        # access complexity of O(1) which is what we want in the next step.
        self.translation_lookup = {}
        for i in range(len(sorted_uniques_values)):
            self.translation_lookup[sorted_uniques_values[i]] = i + 1

    def apply_translation(self, ipt):
        new_matrix = np.zeros(ipt[DataSetTypes.IMAGES]['input'].shape, dtype=np.int64)
        for v in range(len(ipt[DataSetTypes.IMAGES]['input'])):
            if ipt[DataSetTypes.IMAGES]['input'][v] != '0':
                new_matrix[v] = self.translation_lookup[ipt[DataSetTypes.IMAGES]['input'][v]]

        # Swap array
        ipt[DataSetTypes.IMAGES]['input_str'] = ipt[DataSetTypes.IMAGES]['input']
        ipt[DataSetTypes.IMAGES]['input'] = new_matrix

        return ipt
