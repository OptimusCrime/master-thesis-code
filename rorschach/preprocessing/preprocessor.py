#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import re
import shutil

from rorschach.common import DataSetTypes
from rorschach.preprocessing.creators import InputSetCreator, LetterSetCreator
from rorschach.utilities import Config, Filesystem, LoggerWrapper


class Preprocessor:

    OUTPUT_FORMAT = re.compile("^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-[a-z]{6}")

    def __init__(self):
        self.log = LoggerWrapper.load(__name__)

        self.letter_set = None
        self.training_set = None
        self.test_set = None
        self.verification_set = None

    def run(self):
        if Config.get('preprocessing.run') and Config.get('preprocessing.wipe'):
            Preprocessor.wipe_data()

        if Config.get('preprocessing.run'):
            self.create_sets()

        self.save_sets()
        self.save_sets_content()

    @staticmethod
    def wipe_data():
        data_path = Filesystem.get_root_path('data')
        dir_content = os.listdir(data_path)

        for content in dir_content:
            if Preprocessor.OUTPUT_FORMAT.match(content):
                continue

            full_path = os.path.join(data_path, content)
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
                continue

            if content != '.gitignore':
                os.remove(full_path)

    def create_sets(self):
        # Create the data set
        self.log.info('Creating letter set.')
        self.letter_set = LetterSetCreator(DataSetTypes.LETTER_SET)
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

        # Create the word set
        self.log.info('Creating verification set.')
        self.verification_set = InputSetCreator(DataSetTypes.VERIFICATION_SET)
        self.verification_set.generate_random_words()
        self.verification_set.letter_matrices = self.letter_set.contents
        self.verification_set.create()

    def save_sets(self):
        self.letter_set.save()
        self.training_set.save()
        self.test_set.save()
        self.verification_set.save()

    def save_sets_content(self):
        image_set_list = [self.letter_set, self.training_set, self.test_set, self.verification_set]
        for image_set in image_set_list:
            with open(Preprocessor.set_content_file_name(image_set), 'w') as outfile:
                json.dump(image_set.terms, outfile)

    @staticmethod
    def set_content_file_name(image_set):
        file_name = 'verification.json'
        if image_set.type == DataSetTypes.LETTER_SET:
            file_name = 'letter.json'
        if image_set.type == DataSetTypes.TRAINING_SET:
            file_name = 'training.json'
        if image_set.type == DataSetTypes.TEST_SET:
            file_name = 'test.json'

        return Config.get_path('path.output', file_name, fragment=Config.get('uid'))
