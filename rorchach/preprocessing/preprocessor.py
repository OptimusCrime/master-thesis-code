#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil

from rorchach.preprocessing.creators import (DataSetCreator, PhraseCreator,
                                             WordSetCreator)
from rorchach.utilities import Config, Filesystem, LoggerWrapper, unpickle_data


class Preprocessor:

    def __init__(self):
        self.log = LoggerWrapper.load(__name__)

        self.data_set = None
        self.phrase = None
        self.word_set = None

    def run(self):
        if Config.get('preprocessing.run') and Config.get('preprocessing.wipe'):
            Preprocessor.wipe_data()

        if Config.get('preprocessing.run'):
            self.create_sets()

        if self.check_create_phrase():
            self.create_phrase()

        self.save_sets()

    @staticmethod
    def wipe_data():
        data_path = Filesystem.get_root_path('data/')
        dir_content = os.listdir(data_path)

        for content in dir_content:
            full_path = os.path.join(data_path, content)
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
                continue

            if content != '.gitignore':
                os.remove(full_path)

    def create_sets(self):
        # Create the data set
        self.log.info('Creating data set.')
        self.data_set = DataSetCreator()
        self.data_set.create()

        # Create the word set
        self.log.info('Creating word set.')
        self.word_set = WordSetCreator()
        self.word_set.generate_random_words()
        self.word_set.letter_matrices = self.data_set.contents
        self.word_set.create()

    def check_create_phrase(self):
        if Config.get('preprocessing.run'):
            return True

        # Check if we should run just the phrase instead of creating a whole new data set
        if Config.get('preprocessing.new-phrase-run'):
            return Preprocessor.phrase_differs()

    def create_phrase(self):
        # Create the phrase
        self.log.info('Creating phrase set.')
        self.phrase = PhraseCreator()
        self.phrase.terms.append(Config.get('general.phrase'))
        self.phrase.create()

    @staticmethod
    def phrase_differs():
        # If we have phrase file we have to create one either way
        phrase_file = Filesystem.get_root_path('data/phrase.pickl')
        if not os.path.isfile(phrase_file):
            return True

        # If we have a file, check if the current phrase has a new phrase than the one in the config
        phrase_data = unpickle_data(phrase_file)
        if type(phrase_data) is list and len(phrase_data) > 0 and 'text' in phrase_data[0]:
            return phrase_data[0]['text'] != Config.get('general.phrase')

        # If we got here the phrase data is somehow corrupt and we have to create a new one either way
        return True

    def save_sets(self):
        if self.word_set is not None:
            self.data_set.save()

        if self.word_set is not None:
            self.word_set.save()

        if self.phrase is not None:
            self.phrase.save()
