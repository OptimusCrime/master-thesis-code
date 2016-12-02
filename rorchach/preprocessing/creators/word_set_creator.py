#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorchach.preprocessing.creators import TermCreator
from rorchach.preprocessing.savers import MatrixSaver
from rorchach.utilities import Config, Filesystem, pickle_data
from rorchach.wordbuilder.parser import ListParser


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
            if Config.get('preprocessing.word.hardcoded') is not None and \
                    len(Config.get('preprocessing.word.hardcoded')) > 0:
                self.terms.append(Config.get('preprocessing.word.hardcoded'))
            else:
                # Get a random word from our word set
                random_word = self.list_parser.random_word()

                if Config.get('preprocessing.word.max-length') is not None:
                    if len(random_word) > Config.get('preprocessing.word.max-length'):
                        random_word = random_word[0:Config.get('preprocessing.word.max-length')]

                self.terms.append(random_word)

    def save(self):
        if Config.get('preprocessing.save.words-signtures'):
            MatrixSaver.save('data/words-signatures', self.contents)

        pickle_data(self.contents, Filesystem.get_root_path('data/word_set.pickl'))
