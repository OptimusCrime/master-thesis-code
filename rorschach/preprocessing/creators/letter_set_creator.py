#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.preprocessing.creators import AbstractCreator
from rorschach.preprocessing.handlers import TextCreator
from rorschach.preprocessing.savers import MatrixSaver
from rorschach.utilities import Config, Filesystem, pickle_data


class LetterSetCreator(AbstractCreator):

    def create_sets(self):
        contents = []
        for character in Config.get('general.characters'):
            character_arr = TextCreator.write(character)

            self.constraint_handler.calculate(character_arr)

            contents.append({
                'text': character,
                'matrix': character_arr
            })

        # We save the constraints for the letters here. This constraint is used to calculate the upper and lower
        # cutoffs for the phrase and the word set
        self.constraint_handler.save()

        return contents

    def apply_constraints(self):
        for i in range(len(self.contents)):
            # Take the constraints and slice the numpy array. Slicing is [from]:[to] along first the y, then the x axis
            c = self.constraint_handler.constraints

            self.contents[i]['matrix'] = self.contents[i]['matrix'][c['top']:c['bottom'], c['left']:c['right']]

        # Save cropped letters
        if Config.get('preprocessing.save.letters'):
            MatrixSaver.save('data/letters', self.contents)

    def apply_signature(self):
        super().apply_signature()

        # NOTE: We have done the following:
        # 1. Cropped the image to fit perfectly into the ideal letter box
        # 2. Applied the signature crop
        # What we need to do now is to apply a new crop that removes whitespace on the left and right edges to avoid
        # label matcher from wrongfully match letters in words

        for i in range(len(self.contents)):
            edge_left = None
            edge_right = None
            dim = self.contents[i]['matrix'].shape

            for line in self.contents[i]['matrix']:
                for x in range(dim[1]):
                    if line[x] == 0:
                        if edge_left is None or x < edge_left:
                            edge_left = x
                        if edge_right is None or x > edge_right:
                            edge_right = x

            # We slice the matrix here. We do not care about the upper/lower bounds, only the horizontal ones. Note
            # that we had, for some reason, to add + 1 on the right edge. ??? Magic
            self.contents[i]['matrix'] = self.contents[i]['matrix'][::, edge_left:edge_right + 1]

    def save(self):
        # Save signature letters
        if Config.get('preprocessing.save.letters-signatures'):
            MatrixSaver.save('data/letters-signatures', self.contents)

        pickle_data(self.contents, Filesystem.get_root_path('data/letter_set.pickl'))
