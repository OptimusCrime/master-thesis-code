# -*- coding: utf-8 -*-

from rorschach.utilities import Config, Filesystem, pickle_data


class ConstraintHandler:

    def __init__(self):
        self._constraints = None
        self.reset()

    @property
    def constraints(self):
        return {
            'left': self._constraints['left'],
            'right': self._constraints['right'] + 1,
            'top': self._constraints['top'],
            'bottom': self._constraints['bottom'],
        }

    def reset(self):
        self._constraints = {
            'left': None,
            'right': None,
            'top': None,
            'bottom': None
        }

    def save(self):
        pickle_data(self.constraints, Filesystem.save(Config.get('path.data'), 'constraints.pickl'))

    def calculate(self, arr):
        height = len(arr)
        width = len(arr[0])

        # Loop the matrix on both axes, finding their very edge
        for i in range(height):
            for j in range(width):
                if arr[i][j] == 0:
                    # left
                    if self._constraints['left'] is None or \
                            self._constraints['left'] > j:
                        self._constraints['left'] = j

                    # right
                    if self._constraints['right'] is None or \
                            self._constraints['right'] < j:
                        self._constraints['right'] = j

                    # top
                    if self._constraints['top'] is None or \
                            self._constraints['top'] > i:
                        self._constraints['top'] = i

                    # bottom
                    if self._constraints['bottom'] is None or \
                            self._constraints['bottom'] < i:
                        self._constraints['bottom'] = i
