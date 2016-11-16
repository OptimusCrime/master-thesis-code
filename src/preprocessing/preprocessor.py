#!/usr/bin/env python
# -*- coding: utf-8 -*-

from preprocessing.creators import DataSetCreator, PhraseCreator, WordSetCreator
from preprocessing.handlers import PadHandler
from utilities import Config, LoggerWrapper


class Preprocessor:

    def __init__(self):
        self.log = LoggerWrapper.load(__name__)

        self.data_set = None
        self.phrase = None
        self.word_set = None

    def run(self):
        self.create_sets()

        if Config.get('preprocessing.mode') == 'words':
            self.create_word_set()

        if Config.get('preprocessing.pad'):
            self.apply_pad()

        self.save_sets()

    def create_sets(self):
        # Create the data set
        self.log.info('Creating data set.')
        self.data_set = DataSetCreator()
        self.data_set.create()

        # Create the phrase
        self.log.info('Creating phrase set.')
        self.phrase = PhraseCreator()
        self.phrase.terms.append(Config.get('general.phrase'))
        self.phrase.create()

    def create_word_set(self):
        self.log.info('Creating word set.')
        self.word_set = WordSetCreator()
        self.word_set.generate_random_words()
        # TODO BROKEN
        self.word_set.letter_matrices = self.data_set.contents
        self.word_set.create()

    def apply_pad(self):
        self.log.info('Applying padding to sets.')
        pad_handler = PadHandler()
        pad_handler.add(self.phrase.contents)

        if self.word_set is not None:
            pad_handler.add(self.word_set.contents)

        pad_handler.run()

    def save_sets(self):
        self.data_set.save()
        self.phrase.save()

        if self.word_set is not None:
            self.word_set.save()
