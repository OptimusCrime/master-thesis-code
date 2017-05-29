# -*- coding: utf-8 -*-

import copy
import random

import numpy as np

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
        self.values = 0
        self.original = []
        self.randomized = []

    def list_handler(self, input_list, key):
        # Ignore the data set
        if key == DataSetTypes.LETTER_SET:
            return

        super().list_handler(input_list, key)

    def obj_handler(self, obj):
        force_randomness = False
        if Config.get('general.mode') == 'predict' and Config.get('general.special') == 'context':
            if Config.get('general.special-identifier') is not None and\
                    Config.get('general.special-identifier') == obj[DataSetTypes.LABELS]['text']:
                force_randomness = True

        matrix = obj[DataSetTypes.IMAGES]['matrix'][0]
        original_temp = copy.copy(obj[DataSetTypes.IMAGES]['matrix'][0])
        self.values += len(matrix)
        this_randomized_times = 0
        for i in range(len(matrix)):
            if random.randint(0, 100) <= self.random_factor:
                this_randomized_times += 1
                self.randomized_times += 1
                matrix[i] = True if random.randint(0, 1) == 1 else False

        if force_randomness and this_randomized_times == 0:
            print('No randomness added to forced word, running again')
            return self.obj_handler(obj)

        self.original.append(original_temp)

        if force_randomness:
            actual_differences = np.sum(np.not_equal(self.original[-1], obj[DataSetTypes.IMAGES]['matrix'][0]))
            print('Actual differences: ', actual_differences)

        self.randomized.append(copy.copy(obj[DataSetTypes.IMAGES]['matrix'][0]))

        return obj

    def finish(self):
        differences = 0
        for i in range(len(self.original)):
            for j in range(len(self.original[i])):
                if self.original[i][j] != self.randomized[i][j]:
                    differences += 1

        differences_in_percent = (differences / float(self.values)) * 100

        print('Randomizer called', self.randomized_times, 'times.')
        print(differences, 'differences found (', differences_in_percent, '%).')
