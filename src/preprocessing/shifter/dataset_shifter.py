#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utilities.character_handling import CharacterHandling
from utilities.config import Config

import numpy as np


class DatasetShifter:

    def __init__(self, data_set=None, width=None):
        self.data_set = data_set
        self.width = width
        self.labels = None

        self.transformed_index = 0
        self.training_images_transformed = None
        self.training_labels_transformed = None

    def shift(self):
        assert(self.width is not None)

        self.initialize_transformed_sets()

        for i in range(len(self.data_set)):
            loop_length = self.width - len(self.data_set[i]['matrix'][0]) + 1
            matrix = self.data_set[i]['matrix'][0]
            char_index = CharacterHandling.char_to_index(self.data_set[i]['character'])

            self.expand_data_set(loop_length, char_index, matrix)

        self.training_images_transformed = self.training_images_transformed.reshape(
            (len(self.training_images_transformed), 1, self.width, 1))

    def calculate_data_set_size(self):
        size = 0
        for obj in self.data_set:
            size += self.width - len(obj['matrix'][0]) + 1
        return size

    def initialize_transformed_sets(self):
        data_set_size = self.calculate_data_set_size()

        self.training_images_transformed = np.ones((data_set_size, self.width))
        self.training_labels_transformed = np.zeros((data_set_size, len(Config.get('general.characters'))))

    def expand_data_set(self, length, char_index, matrix):
        for j in range(length):
            self.training_labels_transformed[self.transformed_index][char_index] = 1

            # np.arrange arranges from start to finish. In this case we always want it to arrange the length of
            # the matrix, but starting at increasing indexes
            np.put(self.training_images_transformed[self.transformed_index], np.arange(j, (len(matrix) + j)),
                   matrix)

            # Varying indeces between the matrices means that we need to keep the actual index value in a separate
            # variable
            self.transformed_index += 1
