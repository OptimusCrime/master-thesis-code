# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler


class ReshapeSimpleHandler(BaseHandler):

    def __init__(self):
        super().__init__()

        self.highest = 0
        self.find_highest = True

    def run(self, input_lists):
        super().run(input_lists)

        self.find_highest = False

        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        if key == DataSetTypes.LETTER_SET:
            return

        if self.find_highest:
            self.find_highest_embedding(input_list)
            return

        new_list = {DataSetTypes.IMAGES: self.reshape_input(input_list),
                    DataSetTypes.LABELS: self.reshape_labels(input_list)}

        return new_list

    def find_highest_embedding(self, input_list):
        for obj in input_list:
            for value in obj[DataSetTypes.IMAGES]['input']:
                if value > self.highest:
                    self.highest = value

    def reshape_input(self, input_list):
        raw_array = []
        for obj in input_list:
            raw_array.append(self.reshape_actual_input(obj[DataSetTypes.IMAGES]['input']))

        np_array = np.array(raw_array)
        return np_array

    def reshape_actual_input(self, ipt):
        new_arr = np.zeros((len(ipt), self.highest + 1))
        for i in range(len(ipt)):
            new_arr[i][ipt[i]] = 1.

        return new_arr

    def reshape_labels(self, labels_list):
        raw_array = []
        for obj in labels_list:
            raw_array.append(obj[DataSetTypes.LABELS]['value'])

        return np.array(raw_array)
