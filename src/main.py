#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from common import PredictorWrapper
from nets.recurrent import RNNPredictor
from preprocessing import Preprocessor
from utilities import Config, Filesystem, PredictorImporter


def main():
    if Config.get('preprocessing.run'):
        preprocessor = Preprocessor()
        preprocessor.run()

    if Config.get('predicting.run') and Config.get('predicting.predictor') is not None:
        wrapper = PredictorWrapper()
        wrapper.predictor = PredictorImporter.load(Config.get('predicting.predictor'))
        wrapper.run()

    # word_builder = WordBuilder()
    # word_builder.build()
    # word_builder.calculate(predictor.predictions)


if __name__ == '__main__':
    main()
