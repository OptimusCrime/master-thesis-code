# -*- coding: utf-8 -*-

import json

import numpy as np

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler
from rorschach.transformation.handlers.output import IntegerifyLabelHandler
from rorschach.utilities import Config, Filesystem, pickle_data


'''
IntegerifyUniqueLabelHandler

Convert labels into integers while tacking into account unique signatures. This means that characters that have the
very same signature sequence are "added" into one value in the output. Turns the output into a one hot vector.

'''


class IntegerifyUniqueLabelHandler(BaseHandler):

    def __init__(self):
        super().__init__()

        self.label_length = None
        self.label_lookup = {}

    def run(self, input_lists):
        super().run(input_lists)

        self.label_length = IntegerifyLabelHandler.calculate_label_length(input_lists)
        self.build_lookup_table()

        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        if len(self.label_lookup) == 0:
            return input_list

        # Ignore the data set
        if key == DataSetTypes.LETTER_SET:
            return

        super().list_handler(input_list, key)

        return input_list

    def build_lookup_table(self):
        characters_set = []
        for obj in self.input_lists[DataSetTypes.LETTER_SET]:
            characters_set.append(obj[DataSetTypes.IMAGES])

        # Gets list of duplicates [['A', 'B'], ['C'], ['D', 'E', 'F']]
        label_uniques = IntegerifyUniqueLabelHandler.find_label_uniques(characters_set)

        # Save the labels
        pickle_data(label_uniques, Filesystem.save(Config.get('path.data'), 'labels.pickl'))

        with open(Filesystem.save(Config.get('path.data'), 'labels.json'), 'w') as outfile:
            json.dump(label_uniques, outfile)

        # Loop each character, then check if the current character has duplicates. If it has duplicates, check if the
        # other characters are already in the label_lookup table. If it is, use their value instead.
        label_offset = 1
        characters = Config.get('general.characters')

        for i in range(len(characters)):
            labels = IntegerifyUniqueLabelHandler.find_label_unique_count(label_uniques, characters[i])
            if len(labels) == 1:
                self.label_lookup[characters[i]] = label_offset
                label_offset += 1
                continue

            index = self.find_duplicate_lookup_index(labels)
            if index is None:
                self.label_lookup[characters[i]] = label_offset
                label_offset += 1
                continue

            self.label_lookup[characters[i]] = index

        with open(Filesystem.save(Config.get('path.data'), 'labels_lookup.json'), 'w') as outfile:
            json.dump(self.label_lookup, outfile)

    def find_duplicate_lookup_index(self, labels):
        for label in labels:
            if label in self.label_lookup:
                return self.label_lookup[label]
        return None

    @staticmethod
    def find_label_unique_count(label_uniques, char):
        for label in label_uniques:
            if char in label:
                return label
        return None

    @staticmethod
    def find_label_uniques(characters_set):
        # Find which arrays are identical
        label_uniques = {}
        for obj in characters_set:
            matrix = hash(str(obj['matrix']))
            if matrix in label_uniques:
                label_uniques[matrix].append(obj['text'])
                continue

            label_uniques[matrix] = [obj['text']]

        # Turn hash map into list
        char_arr = []
        for key, val in label_uniques.items():
            char_arr.append(val)

        return char_arr

    def obj_handler(self, obj):
        return self.apply_lookup(obj)

    def apply_lookup(self, obj):
        label_matrix = np.zeros(self.label_length, dtype=np.int32)

        for i in range(len(obj[DataSetTypes.IMAGES]['text'])):
            current_char = obj[DataSetTypes.IMAGES]['text'][i]
            char_index = self.label_lookup[current_char]
            label_matrix[i] = char_index

        obj[DataSetTypes.LABELS]['value'] = label_matrix

        return obj
