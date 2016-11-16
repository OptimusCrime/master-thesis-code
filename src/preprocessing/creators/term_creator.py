#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from preprocessing.creators import AbstractCreator
from preprocessing.handlers import TextCreator
from utilities import Filesystem, unpickle_data


class TermCreator(AbstractCreator):

    def __init__(self):
        super().__init__()

        self.terms = []

    def create_sets(self):
        for term in self.terms:
            phrase_image = TextCreator.write(term)
            phrase_arr = np.fromiter(list(phrase_image.getdata()), dtype="int").reshape((phrase_image.height,
                                                                                         phrase_image.width))

            self.constraint_handler.calculate(phrase_arr)

            self.contents.append({
                'text': term,
                'matrix': phrase_arr
            })

    def apply_constraints(self):
        # Load the constraints for upper/lower lines
        char_constraints = unpickle_data(Filesystem.get_root_path('data/constraints.pickl'))

        # Word constraint
        term_constraints = self.constraint_handler.constraints
        for i in range(len(self.contents)):
            self.contents[i]['matrix'] = self.contents[i]['matrix'][
                                         char_constraints['top']:char_constraints['bottom'],
                                         term_constraints['left']:term_constraints['right']]
