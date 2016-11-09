#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from common import PredictorWrapper
from nets.lstm import RNNPredictor
from preprocessing import Preprocessor
from utilities import Config, Filesystem, PredictorImporter


def main():
    if Config.get('force') or not os.path.exists(Filesystem.get_root_path('data/data_set.pickl')):
        Preprocessor.run()

    if Config.get('predictor') is not None:
        wrapper = PredictorWrapper()
        wrapper.predictor = PredictorImporter.load(Config.get('predictor'))
        wrapper.run()

    # word_builder = WordBuilder()
    # word_builder.build()
    # word_builder.calculate(predictor.predictions)


if __name__ == '__main__':
    main()
