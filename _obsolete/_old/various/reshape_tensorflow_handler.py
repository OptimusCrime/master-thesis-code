# !/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler
from rorschach.utilities import Filesystem, pickle_data


class ReshapeTensorflowHandler(BaseHandler):

    def __init__(self):
        super().__init__()

    def run(self, input_lists):
        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        if key != DataSetTypes.TRAINING_SET:
            return

        new_list = {
            DataSetTypes.IMAGES: None,
            DataSetTypes.LABELS: None,
            'voc_size_labels': 19,
            'voc_size_input': None
        }

        new_list[DataSetTypes.IMAGES] = self.reshape_input(input_list)
        new_list[DataSetTypes.LABELS] = self.reshape_labels(input_list)

        voc_size = 0
        for seq in new_list[DataSetTypes.IMAGES]:
            for val in seq:
                if val > voc_size:
                    voc_size = val

        new_list['voc_size_input'] = voc_size + 1

        pickle_data(new_list, Filesystem.get_root_path('tensorflow.pickl'))
        sys.exit()

        return new_list

    def reshape_input(self, input_list):
        raw_array = []
        for obj in input_list:
            raw_array.append(obj[DataSetTypes.IMAGES]['input'])

        np_array = np.array(raw_array)

        return np_array

    def reshape_actual_input(self, ipt):
        return ipt

    def reshape_labels(self, labels_list):
        raw_array = []
        for obj in labels_list:
            inner_arr = np.zeros((len(obj[DataSetTypes.LABELS]['value'], )), dtype='int32')
            for i in range(len(obj[DataSetTypes.LABELS]['value'])):
                inner_arr[i] = np.argmax(obj[DataSetTypes.LABELS]['value'][i])

            raw_array.append(inner_arr)

        return np.array(raw_array)

    def reshape_labels_actual(self, labels):
        inner_arr = [0] * len(labels)
        for i in range(len(labels)):
            inner_arr[i] = np.argmax(labels[i])
        return inner_arr
