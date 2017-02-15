#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

from rorschach.utilities import Config


class SetInvestigator:

    def __init__(self):
        self.sets = []
        self.duplicates = []

    def run(self):
        self.set_information()

        print('')
        print('')

        self.set_duplicates()

    def set_information(self):
        for set_name in self.sets:
            current_set = SetInvestigator.load_file(set_name)

            size = len(current_set)
            unique_words = SetInvestigator.unique_words(current_set)

            print('Current set:       {}'.format(set_name))
            print('Size:              {}'.format(size))
            print('Longest word:      {}'.format(SetInvestigator.longest_word(current_set)))
            print('Average length:    {:.3f}'.format(SetInvestigator.average_length(current_set)))
            print('Unique words:      {}'.format(unique_words))
            print('Duplicate words:   {}'.format(size - unique_words))
            print('----------------------------------')

    def set_duplicates(self):
        for (original, other) in self.duplicates:
            print('Elements in {} that are also in {}'.format(original, other))

            original_set = set(SetInvestigator.load_file(original))
            other_set = set(SetInvestigator.load_file(other))
            duplicates = len(original_set.intersection(other_set))

            print('Duplicates found: {}'.format(duplicates))
            print('----------------------------------')

    @staticmethod
    def average_length(data_set):
        word_length = 0
        for word in data_set:
            word_length += len(word)

        return word_length / len(data_set)

    @staticmethod
    def unique_words(data_set):
        return len(set(data_set))

    @staticmethod
    def longest_word(data_set):
        longest = 0
        for word in data_set:
            if len(word) > longest:
                longest = len(word)
        return longest

    @staticmethod
    def load_file(file_name):
        with open(Config.get_path('path.data', file_name)) as data_file:
            return json.load(data_file)


if __name__ == '__main__':
    set_investigator = SetInvestigator()

    # Regular info
    set_investigator.sets = [
        'training.json',
        'validate.json',
        'test.json'
    ]

    # Duplicates from one set to another
    set_investigator.duplicates = [
        ('training.json', 'validate.json'),
        ('training.json', 'test.json'),
        ('validate.json', 'test.json')
    ]

    # Run
    set_investigator.run()
