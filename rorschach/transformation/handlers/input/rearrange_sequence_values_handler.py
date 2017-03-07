#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

import numpy as np

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler
from rorschach.utilities import Config


class RearrangeSequenceValuesHandler(BaseHandler):

    def __init__(self):
        super().__init__()

        self.bounds = {
            'upper': None,
            'lower': None
        }
        self.rearrange = False

        self.rearranged_pre = {}
        self.rearranged_post = {}

    def run(self, input_lists):
        super().run(input_lists)

        self.dump_content()

        self.rearrange = True

        super().run(input_lists)

        self.dump_content()

        return input_lists

    def list_handler(self, input_list, key):
        # Ignore the data set
        if key == DataSetTypes.LETTER_SET:
            return

        super().list_handler(input_list, key)

        # Add to list so we can dump it as json
        rearrange_list = self.rearranged_pre
        if self.rearrange:
            rearrange_list = self.rearranged_post

        data = []
        for obj in input_list:
            data.append(obj[DataSetTypes.IMAGES]['input'].tolist())

        rearrange_list[key] = data

        return input_list

    def obj_handler(self, obj):
        if self.rearrange:
            return self.rearrange_values(obj)

        self.find_bounds(obj)

        return obj

    def rearrange_values(self, obj):
        if 'null_shift' not in self.data:
            self.data['null_shift'] = abs(self.bounds['lower'])

        input_sequence = obj[DataSetTypes.IMAGES]['input']
        new_matrix = np.zeros(input_sequence.shape, dtype=np.int64)
        for i in range(len(input_sequence)):
            if input_sequence[i] != 0:
                new_matrix[i] = input_sequence[i] + abs(self.bounds['lower'])

        # Swap array
        obj[DataSetTypes.IMAGES]['input_unrearranged'] = obj[DataSetTypes.IMAGES]['input']
        obj[DataSetTypes.IMAGES]['input'] = new_matrix

        return obj

    def find_bounds(self, obj):
        for val in obj[DataSetTypes.IMAGES]['input']:
            if self.bounds['lower'] is None or val < self.bounds['lower']:
                self.bounds['lower'] = val - 1
            if self.bounds['upper'] is None or val > self.bounds['upper']:
                self.bounds['upper'] = val

    def dump_content(self):
        # Dump content before and after shift if we are in debug mode
        if Config.get('general.debug'):
            file_ending = 'rearrange_' + ('pre' if not self.rearrange else 'post') + '.json'
            file_name = Config.get_path('path.output', file_ending, fragment=Config.get('uid'))

            # Get just the inputs
            data = self.rearranged_pre
            if self.rearrange:
                data = self.rearranged_post

            # Dump to JSON
            with open(file_name, 'w') as outfile:
                json.dump(data, outfile)
