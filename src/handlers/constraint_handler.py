#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.utilities.filesystem import Filesystem
from src.utilities.pickler import pickle_data

import numpy as np


class ConstraintHandler:

    def __init__(self):
        self.constraints = {
            'left': {
                'value': None,
                'letter': None
            },
            'right': {
                'value': None,
                'letter': None
            },
            'top': {
                'value': None,
                'letter': None
            },
            'bottom': {
                'value': None,
                'letter': None
            }
        }


    def save(self):
        clean_constraint = {
            'top': self.constraints['top']['value'],
            'bottom': self.constraints['bottom']['value']
        }

        pickle_data(clean_constraint, Filesystem.get_root_path('dump/data/constraints.pickl'))

    def combine(self, old_constraint):
        self.constraints['top']['value'] = old_constraint['top']
        self.constraints['bottom']['value'] = old_constraint['bottom']

    def transform_and_apply(self, image, character):
        # Transform image into numpy matrix
        arr = np.fromiter(list(image.getdata()), dtype="int").reshape((-1, image.width))

        # We just call our centering handler, because we need to do the same later
        self.apply(arr, character)

    def apply(self, arr, key):
        height = len(arr)
        width = len(arr[0])

        # Loop the matrix on both axes, finding their very edge
        for i in range(height):
            for j in range(width):
                if arr[i][j] == 0:
                    # left
                    if self.constraints['left']['value'] is None or \
                            self.constraints['left']['value'] > j:
                        self.constraints['left']['value'] = j
                        self.constraints['left']['letter'] = key

                    # right
                    if self.constraints['right']['value'] is None or \
                            self.constraints['right']['value'] < j:
                        self.constraints['right']['value'] = j
                        self.constraints['right']['letter'] = key

                    # top
                    if self.constraints['top']['value'] is None or \
                            self.constraints['top']['value'] > i:
                        self.constraints['top']['value'] = i
                        self.constraints['top']['letter'] = key

                    # bottom
                    if self.constraints['bottom']['value'] is None or \
                            self.constraints['bottom']['value'] < i:
                        self.constraints['bottom']['value'] = i
                        self.constraints['bottom']['letter'] = key

    def get_constraints(self):
        return tuple([self.constraints['left']['value'],
                      self.constraints['top']['value'],
                      self.constraints['right']['value'],
                      self.constraints['bottom']['value']])
