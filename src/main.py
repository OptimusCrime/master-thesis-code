#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from predicting import PredictorWrapper
from preprocessing import Preprocessor
from utilities import Config, Filesystem
from wordbuilder import WordBuilder


def main():
    if Config.get('force') or not os.path.exists(Filesystem.get_root_path('data/data_set.pickl')):
        Preprocessor.run()

    predictor = PredictorWrapper()
    predictor.run()

    word_builder = WordBuilder()
    word_builder.run()


if __name__ == '__main__':
    main()
