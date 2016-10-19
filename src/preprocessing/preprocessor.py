#!/usr/bin/env python
# -*- coding: utf-8 -*-

from preprocessing.dataset import DataSetCreator
from preprocessing.handlers import SignatureHandler
from preprocessing.phrase import PhraseCreator


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

        # Calculate signature
        signature_handler = SignatureHandler()

        # Apply the signature
        signature_handler.apply(data_set.images)
        signature_handler.apply(phrase.images)

        # Transform and dump
        data_set.transform_and_dump()
        phrase.transform_and_dump()
