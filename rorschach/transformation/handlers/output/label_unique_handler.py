#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from rorschach.common import DataSetTypes
from rorschach.prediction.helpers import PoolingFactorCalculator
from rorschach.transformation.handlers import BaseHandler


class LabelUniqueHandler(BaseHandler):

    def __init__(self):
        super().__init__()

        self.widest_input = None
        self.longest_label = None

        self.label_lookup = {}
        self.label_width = None
        self.label_depth = None

    def run(self, input_lists):
        super().run(input_lists)

        self.calculate_label_width()
        self.calculate_label_depth()

        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        # Ignore the data set
        if key == DataSetTypes.DATA_SET:
            return

        # Require the text and input key
        if len(self.label_lookup) == 0 and len(input_list[DataSetTypes.IMAGES]) > 0:
            assert 'text' in input_list[DataSetTypes.IMAGES][0]
            assert 'input' in input_list[DataSetTypes.IMAGES][0]

        super().list_handler(input_list, key)

    def obj_handler(self, ipt, label):
        if len(self.label_lookup) > 0:
            return self.apply_labels(ipt, label)

        if self.widest_input is None:
            self.widest_input = len(ipt['input'])

        length = len(ipt['text'])
        if self.longest_label is None or length > self.longest_label:
            self.longest_label = length

    def apply_labels(self, ipt, label):
        label_matrix = np.zeros((self.label_width, self.label_depth))
        for i in range(len(ipt['text'])):
            current_char = ipt['text'][i]
            char_index = self.label_lookup[current_char]
            label_matrix[i][char_index] = 1.

        label['value'] = label_matrix

    def calculate_label_width(self):
        results = PoolingFactorCalculator.run(self.widest_input, self.longest_label)
        self.label_width = results['label_width']

    def calculate_label_depth(self):
        characters_set = self.input_lists[DataSetTypes.DATA_SET][DataSetTypes.IMAGES]
        label_uniques = self.find_label_uniques(characters_set)
        self.create_label_lookup_set(characters_set, label_uniques)

    def find_label_uniques(self, characters_set):
        # Find which arrays are identical
        label_uniques = {}
        for obj in characters_set:
            matrix = hash(str(obj['matrix']))
            if matrix in label_uniques:
                label_uniques[matrix].append(obj['text'])
                continue

            label_uniques[matrix] = [obj['text']]

        return label_uniques

    def create_label_lookup_set(self, characters_set, label_uniques):
        # Map from place in the alphabet to correct index in our (soon to be constructed) label matrix. For instance, if
        # A, B and D have the same signature, the output should be something like:
        # A = 0, B = 0, C = 1, D = 0
        current_index = 1
        for i in range(len(characters_set)):
            obj = characters_set[i]
            matrix = hash(str(obj['matrix']))
            matrix_duplicates = label_uniques[matrix]

            if len(matrix_duplicates) == 1:
                # Only one character has this signature, we can just add it to the
                self.label_lookup[obj['text']] = current_index
                current_index += 1
                continue

            # This matrix have duplicates, check if we already have given one of the other duplicates an index
            needle = self.needle_matrix_index(matrix_duplicates)
            if needle is None:
                self.label_lookup[obj['text']] = current_index
                current_index += 1
                continue

            self.label_lookup[obj['text']] = needle

        self.label_depth = current_index

    def needle_matrix_index(self, duplicates):
        for duplicate in duplicates:
            if duplicate in self.label_lookup:
                return self.label_lookup[duplicate]
        return None
