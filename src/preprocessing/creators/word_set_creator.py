#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys

from preprocessing.creators import TermCreator
from preprocessing.savers import MatrixSaver, VerificationSaver
from utilities import Config, CharacterHandling, Filesystem, MatrixDim, pickle_data
from wordbuilder.parser import ListParser


class WordSetCreator(TermCreator):

    def __init__(self):
        super().__init__()

        self.letter_matrices = []
        self.unique_signatures = None
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

        # TODO, crashes with embedding
        if Config.get('preprocessing.save.words-verification') and not Config.get('preprocessing.embedding'):
            VerificationSaver.save('data/words-verification', self.contents)

        pickle_data(self.contents, Filesystem.get_root_path('data/word_set.pickl'))

    def construct_labels(self):
        self.log.info('Constructing word labels matrices.')

        if Config.get('preprocessing.unique-signatures'):
            self.log.info('Removing identical signatures.')
            self.identify_identical_signatures()

        data_set_size = len(self.contents)
        for i in range(data_set_size):
            if (i + 1) % Config.get('logging.batch_reporting') == 0:
                self.log.info('Constructing label matrix for image %s/%s', i + 1, data_set_size)

            word_object = self.contents[i]

            if Config.get('preprocessing.embedding'):
                word_object['label'] = self.construct_embedding_label_matrix(word_object)
            else:
                word_object['label'] = self.construct_label_matrix(word_object)

        self.log.info('Finished constructing word label matrix.')

    def identify_identical_signatures(self):
        unique_signatures_arr = []
        handled = []
        for i in range(len(self.letter_matrices)):
            current_obj = self.letter_matrices[i]
            current_obj['duplicates'] = []
            current_signature = current_obj['matrix']
            current_text = current_obj['text']

            for j in range(i + 1, len(self.letter_matrices)):
                inner_obj = self.letter_matrices[j]
                inner_signature = inner_obj['matrix']
                inner_text = inner_obj['text']

                if np.array_equal(current_signature, inner_signature):
                    current_obj['duplicates'].append(inner_text)
                    handled.append(inner_text)

            if current_text not in handled:
                handled.append(current_text)
                unique_signatures_arr.append(current_obj)

        self.define_unique_indexes(unique_signatures_arr)

    def define_unique_indexes(self, arr):
        self.unique_signatures = {}

        for i in range(len(arr)):
            self.unique_signatures[arr[i]['text']] = i

            for dup in arr[i]['duplicates']:
                self.unique_signatures[dup] = i

        pickle_data(len(arr), Filesystem.get_root_path('data/unique_signatures.pickl'))

    def construct_label_matrix(self, word_object):
        label_matrix = np.zeros((len(word_object['matrix'][0]), MatrixDim.get_size()))
        current_offset = 0
        word_matrix_size = len(word_object['matrix'][0])

        for i in range(len(word_object['text'])):
            character = word_object['text'][i]
            char_matrix = self.matrix_for_char(character)

            while True:
                word_sub_matrix = word_object['matrix'][0][current_offset:current_offset + len(char_matrix[0])]
                if np.array_equal(word_sub_matrix, char_matrix[0]):
                    # Mark the label matrix (set the 2nd dimention to 1 for the corresponding character)
                    self.mark_label_matrix(label_matrix, current_offset, len(word_sub_matrix), character)

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

    def construct_embedding_label_matrix(self, word_object):
        label_matrix = np.zeros((len(word_object['embedding']), MatrixDim.get_size()))

        for i in range(len(word_object['embedding'])):
            if i >= len(word_object['labels_raw']):
                break

            raw_label_value = word_object['labels_raw'][i]
            if raw_label_value is not None:
                char_index = self.get_char_index(raw_label_value)
                label_matrix[i][char_index] = 1

        return label_matrix

    def matrix_for_char(self, char):
        for char_object in self.letter_matrices:
            if char == char_object['text']:
                return char_object['matrix']
        return None

    def get_char_index(self, char):
        if Config.get('preprocessing.unique-signatures'):
            return self.char_to_unique_index(char)

        return CharacterHandling.char_to_index(char)

    def mark_label_matrix(self, matrix, start, length, char):
        char_index = self.get_char_index(char)

        for i in range(start, start + length):
            matrix[i][char_index] = 1

    def char_to_unique_index(self, char):
        return self.unique_signatures[char]
