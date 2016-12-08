#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler


class SequenceHandler(BaseHandler):

    def __init__(self):
        super().__init__()

    def run(self, input_lists):
        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        # Require the concatenated_binary key
        if len(input_list[DataSetTypes.IMAGES]) > 0:
            assert 'concatenated_binary' in input_list[DataSetTypes.IMAGES][0]

        super().list_handler(input_list, key)

    def obj_handler(self, ipt, label):
        sequence = []
        for i in range(1, len(ipt['concatenated_binary'])):
            sequence.append(
                ipt['concatenated_binary'][i - 1] + ipt['concatenated_binary'][i]
            )

        if len(sequence) == 0:
            sequence.append(ipt['concatenated_binary'][0])

        ipt['sequence'] = sequence
