# -*- coding: utf-8 -*-

import copy
import random

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

        # Set seed if defined in the config file
        if Config.get('transformation.noise-random-seed') is not None:
            random.seed(Config.get('transformation.noise-random-seed'))

        self.randomized_times = 0
        self.original = []
        self.randomized = []

    def list_handler(self, input_list, key):
        # Ignore the data set
        if key == DataSetTypes.LETTER_SET:
            return

        super().list_handler(input_list, key)

    def obj_handler(self, obj):
        matrix = obj[DataSetTypes.IMAGES]['matrix'][0]
        self.original.append(copy.copy(obj[DataSetTypes.IMAGES]['matrix'][0]))
        for i in range(len(matrix)):
            if random.randint(0, 100) <= self.random_factor:
                self.randomized_times += 1
                matrix[i] = True if random.randint(0, 1) == 1 else False

        self.randomized.append(copy.copy(obj[DataSetTypes.IMAGES]['matrix'][0]))

        return obj

    def finish(self):
        differences = 0
        for i in range(len(self.original)):
            for j in range(len(self.original[i])):
                if self.original[i][j] != self.randomized[i][j]:
                    differences += 1

        differences_in_percent = (differences / float(len(self.original) * self.original[0].shape[0])) * 100

        print('Randomizer called', self.randomized_times, 'times.')
        print(differences, 'differences found (', differences_in_percent, '%).')

