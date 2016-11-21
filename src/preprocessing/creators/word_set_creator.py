#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys

from preprocessing.creators import TermCreator
from preprocessing.savers import MatrixSaver, VerificationSaver
from utilities import Config, CharacterHandling, Filesystem, pickle_data
from wordbuilder.parser import ListParser


class WordSetCreator(TermCreator):

    def __init__(self):
        super().__init__()

        self.letter_matrices = []
        self.list_parser = ListParser()

    def apply_constraints(self):
        super().apply_constraints()

        if Config.get('preprocessing.save.words'):
            MatrixSaver.save('data/words', self.contents)

    def generate_random_words(self):
        for i in range(Config.get('preprocessing.word.number')):
            if Config.get('preprocessing.word.hardcoded') is not None \
                and len(Config.get('preprocessing.word.hardcoded')) > 0:
                self.terms.append(Config.get('preprocessing.word.hardcoded'))
            else:
                # Get a random word from our word set
                self.terms.append(self.list_parser.random_word())

    def save(self):
        if Config.get('preprocessing.save.words-signtures'):
            MatrixSaver.save('data/words-signatures', self.contents)

        self.construct_labels()

        if Config.get('preprocessing.save.words-verification'):
            VerificationSaver.save('data/words-verification', self.contents)

        pickle_data(self.contents, Filesystem.get_root_path('data/word_set.pickl'))

    def construct_labels(self):
        self.log.info('Constructing word labels matrices.')

        data_set_size = len(self.contents)
        for i in range(data_set_size):
            if (i + 1) % Config.get('logging.batch_reporting') == 0:
                self.log.info('Constructing label matrix for image %s/%s', i + 1, data_set_size)

            word_object = self.contents[i]
            word_object['label'] = self.construct_label_matrix(word_object)

        self.log.info('Finished constructing word label matrix.')

    def construct_label_matrix(self, word_object):
        label_matrix = np.zeros((len(word_object['matrix'][0]), len(Config.get('general.characters'))))
        current_offset = 0
        word_matrix_size = len(word_object['matrix'][0])

        for i in range(len(word_object['text'])):
            character = word_object['text'][i]
            char_matrix = self.matrix_for_char(character)

            while True:
                word_sub_matrix = word_object['matrix'][0][current_offset:current_offset + len(char_matrix[0])]
                if np.array_equal(word_sub_matrix, char_matrix[0]):
                    # Mark the label matrix (set the 2nd dimention to 1 for the corresponding character)
                    WordSetCreator.mark_label_matrix(label_matrix, current_offset, len(word_sub_matrix), character)

                    current_offset += len(word_sub_matrix)
                    break
                else:
                    current_offset += 1

                if current_offset >= word_matrix_size:
                    self.log.error('Could not locate letter %s in image.', character)
                    self.log.error('Image dump:')
                    self.log.error(word_object['matrix'])

                    sys.exit(-1)

        return label_matrix

    def matrix_for_char(self, char):
        for char_object in self.letter_matrices:
            if char == char_object['text']:
                return char_object['matrix']
        return None

    @staticmethod
    def mark_label_matrix(matrix, start, length, char):
        char_index = CharacterHandling.char_to_index(char)
        for i in range(start, start + length):
            matrix[i][char_index] = 1
