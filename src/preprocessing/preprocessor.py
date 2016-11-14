#!/usr/bin/env python
# -*- coding: utf-8 -*-

from preprocessing.dataset import DataSetCreator
from preprocessing.handlers import SignatureHandler
from preprocessing.phrase import PhraseCreator
from preprocessing.words import WordSetCreator
from utilities import Config


class Preprocessor:

    def __init__(self):
        pass

    @staticmethod
    def run():
        # Create the data set
        data_set = DataSetCreator()
        data_set.create()

        # Create the phrase
        phrase = PhraseCreator()
        phrase.create()

        # Apply the signature
        SignatureHandler.apply(data_set.images, set_type=SignatureHandler.DATA_SET)
        SignatureHandler.apply(phrase.images, set_type=SignatureHandler.PHRASE)

        # Transform and dump
        data_set.transform_and_dump()
        phrase.transform_and_dump()

        if Config.get('preprocessing-words'):
            word_set = WordSetCreator()
            word_set.letter_matrices = data_set.dump
            word_set.create()

            SignatureHandler.apply(word_set.images, set_type=SignatureHandler.PHRASE)
            word_set.transform_and_dump()
