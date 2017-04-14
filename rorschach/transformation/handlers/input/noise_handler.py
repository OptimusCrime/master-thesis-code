# -*- coding: utf-8 -*-

from random import randint

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler
from rorschach.utilities import Config

'''
NoiseHandler

Adds noise into the data.

'''


class NoiseHandler(BaseHandler):

    def __init__(self):
        super().__init__()

        self.random_factor = Config.get('transformation.noise-random-factor')

    def list_handler(self, input_list, key):
        # Ignore the data set
        if key == DataSetTypes.LETTER_SET:
            return

        super().list_handler(input_list, key)

    def obj_handler(self, obj):
        matrix = obj[DataSetTypes.IMAGES]['matrix'][0]
        for i in range(len(matrix)):
            if randint(0, 100) <= self.random_factor:
                value = True if randint(0, 1) == 1 else False
                matrix[i] = value
        return obj
