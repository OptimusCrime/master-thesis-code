#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta

import numpy as np

from preprocessing.handlers import ConstraintHandler


class AbstractImageSet:
    __metaclass__ = ABCMeta

    def __init__(self):
        self._images = []
        self.constraint_handler = ConstraintHandler()

    @property
    def images(self):
        return self._images

    def transform_and_dump(self):
        transformed = []
        for image in self._images:
            # Construct numpy array and reshape into matrix
            arr = np.fromiter(list(image['object'].getdata()), dtype="int").reshape((-1, image['object'].width))

            # Get the leftmost and rightmost values, so we can cut away all the whitespace
            left_pad = None
            right_pad = None
            for line in arr:
                for x in range(len(line)):
                    if line[x] == 0:
                        if left_pad is None or x < left_pad:
                            left_pad = x
                        if right_pad is None or x > right_pad:
                            right_pad = x

            # Slice! (I have no idea why I had to add + 1)
            transformed.append({
                'character': image['character'],
                'matrix': arr[::, left_pad:right_pad + 1]
            })

        return transformed
