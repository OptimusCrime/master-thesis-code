#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta

import numpy as np

from preprocessing.handlers import ConstraintHandler
from utilities import Config, LoggerWrapper


class AbstractCreator:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.log = LoggerWrapper.load(__name__)

        self.contents = []
        self.constraint_handler = ConstraintHandler()

    def create(self):
        self.create_sets()
        self.apply_constraints()
        self.apply_signature()

    def create_sets(self):
        pass

    def apply_constraints(self):
        pass

    def apply_signature(self):
        signature_position = Config.get('preprocessing.signature.position')
        signature_height = Config.get('preprocessing.signature.height')

        for i in range(len(self.contents)):
            self.contents[i]['matrix'] = self.contents[i]['matrix'][
                                         signature_position:(signature_position + signature_height)]

    def save(self):
        pass

    def transform_and_dump(self):
        transformed = []
        for content in self.contents:
            # Construct numpy array and reshape into matrix
            arr = np.fromiter(list(content['object'].getdata()), dtype="int").reshape((-1, content['object'].width))

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
                'character': content['character'],
                'matrix': arr[::, left_pad:right_pad + 1]
            })

        return transformed
