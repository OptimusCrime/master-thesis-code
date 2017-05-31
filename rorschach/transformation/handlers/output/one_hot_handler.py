# -*- coding: utf-8 -*-

import numpy as np

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler
from rorschach.utilities import Config, unpickle_data


'''
OneHotHandler

Take a vector and turn in into a one hot handler

'''


class OneHotHandler(BaseHandler):

    def __init__(self):
        super().__init__()

        self.depth = len(unpickle_data(Config.get_path('path.data', 'labels.pickl')))

    def list_handler(self, input_list, key):
        # Ignore the data set
        if key == DataSetTypes.LETTER_SET:
            return

        super().list_handler(input_list, key)

        return input_list

    def obj_handler(self, obj):
        new_matrix = np.zeros((len(obj[DataSetTypes.LABELS]['value']), self.depth), dtype=np.int32)
        labels = obj[DataSetTypes.LABELS]['value']
        for i in range(len(labels)):
            new_matrix[i][labels[i]] = 1

        # Swap old and new
        obj[DataSetTypes.LABELS]['value_short'] = labels
        obj[DataSetTypes.LABELS]['value'] = new_matrix

        return obj
