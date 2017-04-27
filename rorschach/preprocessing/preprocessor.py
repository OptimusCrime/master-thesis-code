# -*- coding: utf-8 -*-

import json
import os
import re
import shutil

from rorschach.common import DataSetTypes
from rorschach.preprocessing.creators import InputSetCreator, LetterSetCreator
from rorschach.utilities import Config, Filesystem, LoggerWrapper, pickle_data


class Preprocessor:

    OUTPUT_FORMAT = re.compile("^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-[a-z]{6}")

    def __init__(self):
        self.log = LoggerWrapper.load(__name__)

        self.letter_set = None
        self.training_set = None
        self.validate_set = None
        self.test_set = None

    def run(self):
        if Config.get('preprocessing.run') and Config.get('preprocessing.wipe'):
            Preprocessor.wipe_images()
            Preprocessor.wipe_data()

        if Config.get('preprocessing.run'):
            self.create_sets()

            # Save related information concerning this data set
            Preprocessor.save_information()

        self.save_sets()
        self.save_sets_content()

    @staticmethod
    def wipe_images():
        images_path = Config.get('path.image')
        if os.path.exists(images_path):
            shutil.rmtree(images_path)

    @staticmethod
    def wipe_data():
        data_path = Config.get('path.data')
        if os.path.exists(data_path):
            shutil.rmtree(data_path)

    def create_sets(self):
        # Create the data set
        self.log.info('Creating letter set.')
        self.letter_set = LetterSetCreator(DataSetTypes.LETTER_SET)
        self.letter_set.create()

        # Create the training set
        self.log.info('Creating training set.')
        self.training_set = InputSetCreator(DataSetTypes.TRAINING_SET)
        self.training_set.generate_random_words()
        self.training_set.letter_matrices = self.letter_set.contents
        self.training_set.create()

        # Create the verification set
        self.log.info('Creating validate set.')
        self.validate_set = InputSetCreator(DataSetTypes.VALIDATE_SET)
        self.validate_set.generate_random_words()
        self.validate_set.letter_matrices = self.letter_set.contents
        self.validate_set.create()

        # Create the test set
        self.log.info('Creating test set.')
        self.test_set = InputSetCreator(DataSetTypes.TEST_SET)
        self.test_set.generate_random_words()
        self.test_set.letter_matrices = self.letter_set.contents
        self.test_set.create()

    def save_sets(self):
        self.letter_set.save()
        self.training_set.save()
        self.test_set.save()
        self.validate_set.save()

    def save_sets_content(self):
        image_set_list = [self.letter_set, self.training_set, self.validate_set, self.test_set]
        for image_set in image_set_list:
            with open(Preprocessor.set_content_file_name(image_set), 'w') as outfile:
                json.dump(image_set.terms, outfile)

    @staticmethod
    def save_information():
        pickle_data({
            'label_length': Config.get('preprocessing.input.max-length')
        }, Config.get_path('path.data', 'information.pickl'))

    @staticmethod
    def set_content_file_name(image_set):
        file_name = 'validate.json'
        if image_set.type == DataSetTypes.LETTER_SET:
            file_name = 'letter.json'
        if image_set.type == DataSetTypes.TRAINING_SET:
            file_name = 'training.json'
        if image_set.type == DataSetTypes.TEST_SET:
            file_name = 'test.json'

        return Config.get_path('path.data', file_name)
