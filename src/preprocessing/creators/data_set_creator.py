#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from preprocessing.creators import AbstractCreator
from preprocessing.handlers import TextCreator
from utilities import Config, Filesystem, pickle_data


class DataSetCreator(AbstractCreator):

    def create_sets(self):
        for character in Config.get('general.characters'):
            character_image = TextCreator.write(character)
            character_arr = np.fromiter(list(character_image.getdata()), dtype="int").reshape((character_image.height,
                                                                                               character_image.width))

            self.constraint_handler.calculate(character_arr)

            self.contents.append({
                'text': character,
                'matrix': character_arr
            })

        # We save the constraints for the letters here. This constraint is used to calculate the upper and lower
        # cutoffs for the phrase and the word set
        self.constraint_handler.save()

    def apply_constraints(self):
        if Config.get('preprocessing.save'):
            Filesystem.create('data/crop')

        for i in range(len(self.contents)):
            # Take the constraints and slice the numpy array. Slicing is [from]:[to] along first the y, then the x axis
            c = self.constraint_handler.constraints

            self.contents[i]['matrix'] = self.contents[i]['matrix'][c['top']:c['bottom'], c['left']:c['right']]

            # TODO broken
            if Config.get('preprocessing.save'):
                self.contents[i]['object'].save(Filesystem.get_root_path(
                    'data/crop/' + self.contents[i]['character'] + '.png'))

    def save(self):
        pickle_data(self.contents, Filesystem.get_root_path('data/data_set.pickl'))
