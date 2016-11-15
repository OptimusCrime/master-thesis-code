#!/usr/bin/env python
# -*- coding: utf-8 -*-

from preprocessing.dataset import DataSetCreator
from preprocessing.handlers import SignatureHandler
from preprocessing.phrase import PhraseCreator
from preprocessing.words import WordSetCreator
from utilities import Config, LoggerWrapper


class Preprocessor:

    def __init__(self):
        self.log = LoggerWrapper.load(__name__)

    def run(self):
        # Create the data set
        self.log.info('Creating data set.')
        data_set = DataSetCreator()
        data_set.create()

        # Create the phrase
        self.log.info('Creating phrase set.')
        phrase = PhraseCreator()
        phrase.create()

        # Apply the signature
        self.log.info('Applying signature to data set.')
        SignatureHandler.apply(data_set.images, set_type=SignatureHandler.DATA_SET)

        self.log.info('Applying signature to phrase set.')
        SignatureHandler.apply(phrase.images, set_type=SignatureHandler.PHRASE)

        # Transform and dump
        self.log.info('Transforming and dumping data set.')
        data_set.transform_and_dump()

        self.log.info('Transforming and dumping phrase set.')
        phrase.transform_and_dump()

        if Config.get('preprocessing.mode') == 'words':
            self.log.info('Creating word set.')
            word_set = WordSetCreator()
            word_set.letter_matrices = data_set.dump
            word_set.create()

            self.log.info('Applying signature to word set.')
            SignatureHandler.apply(word_set.images, set_type=SignatureHandler.PHRASE)

            self.log.info('Transforming and dumping word set.')
            word_set.transform_and_dump()
