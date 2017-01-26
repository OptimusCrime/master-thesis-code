# !/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler


class LabelInitializeHandler(BaseHandler):

    def __init__(self):
        super().__init__()

    def run(self, input_lists):
        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        input_length = len(input_list[DataSetTypes.IMAGES])

        # Create empty labels for this data set (avoid index out of bounds later)
        input_list[DataSetTypes.LABELS] = [{}] * input_length
