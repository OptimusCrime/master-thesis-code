#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from preprocessing.abstracts import AbstractImageSet
from preprocessing.handlers import TextCreator
from utilities import Config, CharacterHandling, Filesystem, LoggerWrapper, pickle_data
from wordbuilder.parser import ListParser


class WordSetCreator(AbstractImageSet):

    def __init__(self):
        super().__init__()

        self.log = LoggerWrapper.load(__name__)

        self.letter_matrices = []
        self.list_parser = ListParser()

    def create(self):
        self.create_images()
        self.apply_constraints()

    def create_images(self):
        self.log.info('Prepearing to create %s images for the word set.', Config.get('preprocessing.word.number'))

        for i in range(Config.get('preprocessing.word.number')):
            if (i + 1) % Config.get('logging.batch_reporting') == 0:
                self.log.info('Creating image %s/%s.', i + 1, Config.get('preprocessing.word.number'))

            # Get a random word from our word set
            random_word = self.list_parser.random_word()

            word_image = TextCreator.write(random_word)

            self.constraint_handler.transform_and_apply(word_image, random_word)

            self._images.append({
                'character': random_word,
                'object': word_image
            })

            self.constraint_handler.save()

        self.log.info('Finished creating word image set.')

    def apply_constraints(self):
        if Config.get('preprocessing.save'):
            Filesystem.create('data/words')

        image_set_size = len(self.images)

        self.log.info('Applying constraints to images.')

        for i in range(image_set_size):
            if (i + 1) % Config.get('logging.batch_reporting') == 0:
                self.log.info('Applying constraint to image %s/%s.', i + 1, image_set_size)

            self._images[i]['object'] = self._images[i]['object'].crop(self.constraint_handler.constraints)

            if Config.get('preprocessing.save'):
                self._images[i]['object'].save(Filesystem.get_root_path(
                    'data/words/' + self._images[i]['character'] + '.png'))

    def transform_and_dump(self):
        self.log.info('Transforming word set.')

        # Run parent method. Transform data
        transformed = super(WordSetCreator, self).transform_and_dump()
        data_set_size = len(transformed)

        self.log.info('Done transforming word set.')
        self.log.info('Constructing word label matrix.')

        for i in range(data_set_size):
            if (i + 1) % Config.get('logging.batch_reporting') == 0:
                self.log.info('Constructing label matrix for image %s/%s', i + 1, data_set_size)

            word_object = transformed[i]
            word_object['label'] = self.construct_label_matrix(word_object)

            # Store data for later use
            pickle_data(transformed, Filesystem.get_root_path('data/word_set.pickl'))

        self.log.info('Finished constructing word label matrix.')

    def matrix_for_char(self, char):
        for char_object in self.letter_matrices:
            if char == char_object['character']:
                return char_object['matrix']
        return None

    def construct_label_matrix(self, word_object):
        label_matrix = np.zeros((len(word_object['matrix'][0]), len(Config.get('general.characters'))))
        current_offset = 0
        word_matrix_size = len(word_object['matrix'][0])

        for i in range(len(word_object['character'])):
            character = word_object['character'][i]
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

                if current_offset == word_matrix_size:
                    break

        return label_matrix

    @staticmethod
    def mark_label_matrix(matrix, start, length, char):
        char_index = CharacterHandling.char_to_index(char)
        for i in range(start, start + length):
            matrix[i][char_index] = 1
