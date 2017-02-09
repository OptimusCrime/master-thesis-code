#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil

from rorschach.common import DataSetTypes
from rorschach.preprocessing.creators import InputSetCreator, LetterSetCreator
from rorschach.utilities import Config, Filesystem, LoggerWrapper


class Preprocessor:

    def __init__(self):
        self.log = LoggerWrapper.load(__name__)

        self.letter_set = None
        self.training_set = None
        self.test_set = None

    def run(self):
        if Config.get('preprocessing.run') and Config.get('preprocessing.wipe'):
            Preprocessor.wipe_data()

        if Config.get('preprocessing.run'):
            self.create_sets()

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
        self.log.info('Creating letter set.')
        self.letter_set = LetterSetCreator()
        self.letter_set.create()

        # Create the word set
        self.log.info('Creating training set.')
        self.training_set = InputSetCreator(DataSetTypes.TRAINING_SET)
        self.training_set.generate_random_words()
        self.training_set.letter_matrices = self.letter_set.contents
        self.training_set.create()

        # Create the word set
        self.log.info('Creating test set.')
        self.test_set = InputSetCreator(DataSetTypes.TEST_SET)
        self.test_set.generate_random_words()
        self.test_set.letter_matrices = self.letter_set.contents
        self.test_set.create()

    def save_sets(self):
        self.letter_set.save()
        self.training_set.save()
        self.test_set.save()
