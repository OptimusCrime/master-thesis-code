#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler


class SequenceHandler(BaseHandler):

    def __init__(self):
        super().__init__()

    def obj_handler(self, obj):
        sequence = []
        for i in range(1, len(obj[DataSetTypes.IMAGES]['concatenated_binary'])):
            sequence.append(
                obj[DataSetTypes.IMAGES]['concatenated_binary'][i - 1] + obj[DataSetTypes.IMAGES]['concatenated_binary'][i]
            )

        if len(sequence) == 0:
            sequence.append(obj[DataSetTypes.IMAGES]['concatenated_binary'][0])

        obj[DataSetTypes.IMAGES]['sequence'] = sequence

        return obj
