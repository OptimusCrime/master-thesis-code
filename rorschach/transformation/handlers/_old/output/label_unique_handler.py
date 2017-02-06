#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from prediction.common.helpers import PoolingFactorCalculator
from rorschach.common import DataSetTypes
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
        if key == DataSetTypes.LETTER_SET:
            return

        super().list_handler(input_list, key)

        return input_list

    def obj_handler(self, obj):
        if len(self.label_lookup) > 0:
            return self.apply_labels(obj)

        if self.widest_input is None:
            self.widest_input = len(obj[DataSetTypes.IMAGES]['input'])

        length = len(obj[DataSetTypes.IMAGES]['text'])
        if self.longest_label is None or length > self.longest_label:
            self.longest_label = length

        return obj

    def apply_labels(self, obj):
        label_matrix = np.zeros((self.label_width, self.label_depth + 1))
        for i in range(len(label_matrix)):
            label_matrix[i][0] = 1.

        for i in range(len(obj[DataSetTypes.IMAGES]['text'])):
            current_char = obj[DataSetTypes.IMAGES]['text'][i]
            char_index = self.label_lookup[current_char]
            label_matrix[i][0] = 0.
            label_matrix[i][char_index + 1] = 1.

        obj[DataSetTypes.LABELS]['value'] = label_matrix

        return obj

    def calculate_label_width(self):
        results = PoolingFactorCalculator.run(self.widest_input, self.longest_label)
        self.label_width = results['width_label']

        # TODO SUCH A HACK!!!
        self.label_width = 5

    def calculate_label_depth(self):
        characters_set = []
        for obj in self.input_lists[DataSetTypes.LETTER_SET]:
            characters_set.append(obj[DataSetTypes.IMAGES])

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
