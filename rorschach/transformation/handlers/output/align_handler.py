# -*- coding: utf-8 -*-

import numpy as np

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler
from rorschach.utilities import Config, unpickle_data

'''
AlignHandler

Special handler that pads the output to the same width as the input. Necessary for networks which simply feeds the
sequences into an LSTM and uses the entire return sequence as the output.

'''


class AlignHandler(BaseHandler):

    def __init__(self):
        super().__init__()

        self.length = None
        self.depth = None

    def run(self, input_lists):
        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        # Ignore the data set
        if key == DataSetTypes.LETTER_SET:
            return

        if self.length is None:
            self.calculate_length(input_list)

        super().list_handler(input_list, key)

        return input_list

    def calculate_length(self, input_list):
        if len(input_list) > 0:
            self.length = len(input_list[0][DataSetTypes.IMAGES]['input'])
            self.depth = len(unpickle_data(Config.get_path('path.data', 'labels.pickl')))

    def obj_handler(self, obj):
        new_matrix = np.zeros((self.length, self.depth), dtype=np.int32)
        labels = obj[DataSetTypes.LABELS]['value']
        for i in range(len(labels)):
            if labels[i] == 0:
                break

            new_matrix[i][labels[i]] = 1

        # Swap old and new
        obj[DataSetTypes.LABELS]['value_short'] = labels
        obj[DataSetTypes.LABELS]['value'] = new_matrix

        return obj
