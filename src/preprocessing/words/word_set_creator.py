#!/usr/bin/env python
# -*- coding: utf-8 -*-

from preprocessing.abstracts import AbstractImageSet
from preprocessing.handlers import TextCreator
from utilities import Config, Filesystem, pickle_data, unpickle_data
from wordbuilder.parser import ListParser


class WordSetCreator(AbstractImageSet):

    def __init__(self):
        super().__init__()

        self.letter_matrices = []
        self.list_parser = ListParser()

    def create(self):
        self.create_images()

        print(self._images)
        self.apply_constraints()

    def create_images(self):
        for i in range(2):
            # Get a random word from our word set
            random_word = self.list_parser.random_word()

            word_image = TextCreator.write(random_word)

            self.constraint_handler.transform_and_apply(word_image, random_word)

            self._images.append({
                'character': random_word,
                'object': word_image
            })

            self.constraint_handler.save()

    def apply_constraints(self):
        if Config.get('preprocessing-save'):
            Filesystem.create('data/words')

        for i in range(len(self.images)):
            self._images[i]['object'] = self._images[i]['object'].crop(self.constraint_handler.constraints)

            if Config.get('preprocessing-save'):
                self._images[i]['object'].save(Filesystem.get_root_path(
                    'data/words/' + self._images[i]['character'] + '.png'))
