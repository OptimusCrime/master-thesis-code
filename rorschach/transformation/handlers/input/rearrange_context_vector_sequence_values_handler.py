# -*- coding: utf-8 -*-


import numpy as np

from rorschach.common import Constants, DataSetTypes
from rorschach.transformation.handlers import BaseHandler


'''
RearrangeContextVectorSequenceValuesHandler

Because embedding solutions require whole integer values, we need to rearrange our upper and lower bounds and shift
the values so that all values are whole integers.

Special variant for the context vector plotting

'''


class RearrangeContextVectorSequenceValuesHandler(BaseHandler):

    def __init__(self):
        super().__init__()

        self.size = None

    def run(self, input_lists):
        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        # Ignore the data set
        if key == DataSetTypes.LETTER_SET:
            return

        if self.size is None:
            self.size = len(input_list[0][DataSetTypes.IMAGES]['input'])

        return super().list_handler(input_list, key)

    def obj_handler(self, obj):
        input_sequence = obj[DataSetTypes.IMAGES]['input']
        new_matrix = np.zeros(input_sequence.shape, dtype=np.int64)

        for i in range(len(input_sequence)):
            if input_sequence[i] == 0:
                new_matrix[i] = self.size
                continue

            if input_sequence[i] == Constants.STOP_WORD:
                new_matrix[i] = (self.size * 2) + 2
                continue

            if input_sequence[i] < 0:
                new_matrix[i] = self.size - abs(input_sequence[i])
                continue

            new_matrix[i] = self.size + input_sequence[i]

        # Swap array
        obj[DataSetTypes.IMAGES]['input_unrearranged'] = obj[DataSetTypes.IMAGES]['input']
        obj[DataSetTypes.IMAGES]['input'] = new_matrix

        return obj
