#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler


class EmbeddingHandler(BaseHandler):

    def __init__(self):
        super().__init__()

        self.uniques = {}
        self.embedding_lookup = None

    def run(self, input_lists):
        super().run(input_lists)
        self.create_embedding_set()

        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        # Ignore the data set
        if key == DataSetTypes.LETTER_SET:
            return

        # Require the input key
        if len(input_list[DataSetTypes.IMAGES]) > 0:
            assert 'input' in input_list[DataSetTypes.IMAGES][0]

        super().list_handler(input_list, key)

    def obj_handler(self, ipt, label):
        if self.embedding_lookup is not None:
            return self.apply_embedding(ipt)

        for val in ipt['input']:
            if val == '0':
                continue

            if val in self.uniques:
                self.uniques[val] += 1
            else:
                self.uniques[val] = 1

    def create_embedding_set(self):
        # We are now sorting the values descending by their popularity
        sorted_uniques_values = sorted(self.uniques, key=self.uniques.get, reverse=True)

        # Reassign the dict's values to be the index from the sorting, e.i. their popularity. The list still has a
        # access complexity of O(1) which is what we want in the next step.
        self.embedding_lookup = {}
        for i in range(len(sorted_uniques_values)):
            self.embedding_lookup[sorted_uniques_values[i]] = i + 1

    def apply_embedding(self, ipt):
        new_matrix = np.zeros(ipt['input'].shape, dtype=np.int64)
        for v in range(len(ipt['input'])):
            if ipt['input'][v] != '0':
                new_matrix[v] = self.embedding_lookup[ipt['input'][v]]

        # Swap array
        ipt['input_str'] = ipt['input']
        ipt['input'] = new_matrix
