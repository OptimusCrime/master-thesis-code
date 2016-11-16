#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utilities import Filesystem, pickle_data


class ConstraintHandler:
    def __init__(self):
        self.constraints = {
            'left': None,
            'right': None,
            'top': None,
            'bottom': None
        }

    def save(self):
        pickle_data(self.constraints, Filesystem.get_root_path('data/constraints.pickl'))

    def calculate(self, arr):
        height = len(arr)
        width = len(arr[0])

        # Loop the matrix on both axes, finding their very edge
        for i in range(height):
            for j in range(width):
                if arr[i][j] == 0:
                    # left
                    if self.constraints['left'] is None or \
                            self.constraints['left'] > j:
                        self.constraints['left'] = j

                    # right
                    if self.constraints['right'] is None or \
                            self.constraints['right'] < j:
                        self.constraints['right'] = j

                    # top
                    if self.constraints['top'] is None or \
                            self.constraints['top'] > i:
                        self.constraints['top'] = i

                    # bottom
                    if self.constraints['bottom'] is None or \
                            self.constraints['bottom'] < i:
                        self.constraints['bottom'] = i
