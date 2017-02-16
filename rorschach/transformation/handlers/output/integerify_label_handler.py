#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

import numpy as np

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler
from rorschach.utilities import Config, pickle_data, Filesystem


class IntegerifyLabelHandler(BaseHandler):

    def __init__(self):
        super().__init__()

        self.label_length = None
        self.label_lookup = {}

    def run(self, input_lists):
        self.label_length = IntegerifyLabelHandler.calculate_label_length(input_lists)
        self.build_lookup_table()
        self.store_label_file()

        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        # Ignore the data set
        if key == DataSetTypes.LETTER_SET:
            return

        super().list_handler(input_list, key)

        return input_list

    @staticmethod
    def calculate_label_length(input_lists):
        if Config.get('preprocessing.input.max-length') is not None:
            return Config.get('preprocessing.input.max-length')

        raise NotImplementedError("Damnit")

    def build_lookup_table(self):
        characters = Config.get('general.characters')
        for i in range(len(characters)):
            self.label_lookup[characters[i]] = i + 1

    def store_label_file(self):
        # Get the array of our character. Also add 0 because our words can be padded.
        labels = Config.get('general.characters')
        labels.append('0')

        # Save the labels
        pickle_data(labels, Filesystem.save(Config.get('path.data'), 'labels.pickl'))

        with open(Filesystem.save(Config.get('path.data'), 'labels.json'), 'w') as outfile:
            json.dump(labels, outfile)

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
