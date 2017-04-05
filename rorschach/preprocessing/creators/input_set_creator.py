# -*- coding: utf-8 -*-

import os

from rorschach.common import DataSetTypes
from rorschach.preprocessing.creators import TermCreator
from rorschach.preprocessing.savers import MatrixSaver
from rorschach.utilities import Config, Filesystem, pickle_data
from rorschach.wordlist import WordListParser


class InputSetCreator(TermCreator):

    def __init__(self, type=DataSetTypes.TRAINING_SET):
        super().__init__(type)

        self.letter_matrices = []
        self.unique_signatures = None
        self.word_list_parser = WordListParser()

    def apply_constraints(self):
        super().apply_constraints()

        if Config.get('preprocessing.save.cropped'):
            MatrixSaver.save(os.path.join('cropped', self.set_type_keyword), self.contents)

    def generate_random_words(self):
        for i in range(Config.get('preprocessing.' + self.set_type_keyword + '-set.size')):
            random_word = self.word_list_parser.random_word(word_list=self.type)
            self.terms.append(random_word)

    def save(self):
        if Config.get('preprocessing.save.signatures'):
            MatrixSaver.save(os.path.join('signatures', self.set_type_keyword), self.contents)

        pickle_data(self.contents, Filesystem.save(Config.get('path.data'), self.set_type_keyword + '_set.pickl'))
