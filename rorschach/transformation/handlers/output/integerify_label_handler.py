# -*- coding: utf-8 -*-

import json
import os

import numpy as np

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler
from rorschach.utilities import Config, Filesystem, pickle_data, unpickle_data


'''
IntegerifyLabelHandler

Convert labels into integers ignoring unique signatures.

'''


class IntegerifyLabelHandler(BaseHandler):

    def __init__(self):
        super().__init__()

        self.label_length = None
        self.label_lookup = {}

    def run(self, input_lists):
        self.label_length = IntegerifyLabelHandler.calculate_label_length()
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
    def calculate_label_length():
        information_file = Config.get_path('path.data', 'information.pickl')
        if not os.path.exists(information_file):
            raise Exception('Information file information.pickl not found in data directory')

        data = unpickle_data(Config.get_path('path.data', 'information.pickl'))
        if 'label_length' not in data:
            raise Exception('Label length information not found in information.pickl file')

        return data['label_length']

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

        with open(Filesystem.save(Config.get('path.data'), 'labels_lookup.json'), 'w') as outfile:
            json.dump(self.label_lookup, outfile)

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
