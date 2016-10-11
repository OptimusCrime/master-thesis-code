#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utilities import Filesystem, pickle_data, unpickle_data


class Predicter:

    def __init__(self):
        pass

    def matrices_match(self, phrase, offset, character):
        for i in range(len(character)):
            if character[i] != phrase[offset + i]:
                return False

        return True

    def naive_match(self, phrase, offset, character):
        if len(phrase['matrix'][0]) - offset >= len(character['matrix'][0]):
            # Do stuff
            return self.matrices_match(phrase['matrix'][0], offset, character['matrix'][0])

        return False

    def run(self):
        # Load the phrase
        phrase = unpickle_data(Filesystem.get_root_path('data/phrase.pickl'))[0]

        # Load the data set
        data_set = unpickle_data(Filesystem.get_root_path('data/data_set.pickl'))

        # WARNING: HEAVY WORK IN PROGRESS
        phrase_length = len(phrase['matrix'][0])
        predicts = []
        for i in range(phrase_length):
            for character_set in data_set:
                if (self.naive_match(phrase, i, character_set)):
                    predicts.append({
                        'character': character_set['character'],
                        'offset': i,
                        'matrix': character_set['matrix'][0]
                    })

        print(phrase['matrix'][0])
        print(predicts)

