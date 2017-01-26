#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np

from rorschach.preprocessing.creators import AbstractCreator
from rorschach.preprocessing.handlers import TextCreator
from rorschach.utilities import Config, Filesystem, unpickle_data


class TermCreator(AbstractCreator):

    def __init__(self):
        super().__init__()

        self.terms = []

    def create_sets(self):
        data_set_size = len(self.terms)
        for i in range(data_set_size):
            if (i + 1) % Config.get('logging.batch_reporting') == 0:
                self.log.info('Constructing image %s/%s.', i + 1, data_set_size)

            phrase_image = TextCreator.write(self.terms[i])
            phrase_arr = np.fromiter(list(phrase_image.getdata()), dtype="int").reshape((phrase_image.height,
                                                                                         phrase_image.width))

            self.contents.append({
                'text': self.terms[i],
                'matrix': phrase_arr
            })

    def apply_constraints(self):
        constraints_file = Filesystem.get_root_path('data/constraints.pickl')

        # We can not apply constraints unless we have the pickle file
        assert os.path.isfile(constraints_file) is True

        # Load the constraints for upper/lower lines
        char_constraints = unpickle_data(constraints_file)

        data_set_size = len(self.contents)
        for i in range(data_set_size):
            if (i + 1) % Config.get('logging.batch_reporting') == 0:
                self.log.info('Applying constraints for image %s/%s.', i + 1, data_set_size)

            # Calculate constraints for THIS unique term
            self.constraint_handler.reset()
            self.constraint_handler.calculate(self.contents[i]['matrix'])

            self.contents[i]['matrix'] = self.contents[i]['matrix'][
                                         char_constraints['top']:char_constraints['bottom'],
                                         self.constraint_handler.constraints['left']:
                                         self.constraint_handler.constraints['right']]
