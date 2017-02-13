#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.prediction import PredictorWrapper
from rorschach.preprocessing import Preprocessor
from rorschach.utilities import Config, ModuleImporter, UidGenerator


def main():
    UidGenerator.run()

    if Config.get('preprocessing.run'):
        preprocessor = Preprocessor()
        preprocessor.run()

    if Config.get('predicting.run') and Config.get('predicting.predictor') is not None:
        wrapper = PredictorWrapper()
        wrapper.predictor = ModuleImporter.load(Config.get('predicting.predictor'))
        wrapper.run()


if __name__ == '__main__':
    main()
