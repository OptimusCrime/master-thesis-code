#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.common import DataSetTypes
from rorschach.preprocessing.creators import TermCreator
from rorschach.preprocessing.savers import MatrixSaver
from rorschach.utilities import Config, Filesystem, pickle_data
from rorschach.wordlist import WordListParser


class InputSetCreator(TermCreator):

    def __init__(self, type=DataSetTypes.TRAINING_SET):
        super().__init__()

        self.type = type
        self.letter_matrices = []
        self.unique_signatures = None
        self.word_list_parser = WordListParser()

        self.type_keyword = 'training'
        if self.type == DataSetTypes.TEST_SET:
            self.type_keyword = 'test'

    def apply_constraints(self):
        super().apply_constraints()

        if Config.get('preprocessing.save.' + self.type_keyword):
            MatrixSaver.save('data/' + self.type_keyword, self.contents)

    def generate_random_words(self):
        for i in range(Config.get('preprocessing.' + self.type_keyword + '-set.size')):
            random_word = self.word_list_parser.random_word(Config.get('preprocessing.input.max-length'))
            self.terms.append(random_word)

    def save(self):
        if Config.get('preprocessing.save.' + self.type_keyword + '-signtures'):
            MatrixSaver.save('data/' + self.type_keyword + '-signatures', self.contents)

        pickle_data(self.contents, Filesystem.get_root_path('data/' + self.type_keyword + '_set.pickl'))
