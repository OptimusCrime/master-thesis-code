#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.prediction import PredictorWrapper
from rorschach.preprocessing import Preprocessor
from rorschach.utilities import Config, ModuleImporter, UidGenerator, LoggerWrapper


def main():
    UidGenerator.run()

    log = LoggerWrapper.load(__name__)
    log.info('Current uid is %s', Config.get('uid'))
    log.info('Storing output in %s', Config.get_path('path.output', '', fragment=Config.get('uid')))

    if Config.get('preprocessing.run'):
        preprocessor = Preprocessor()
        preprocessor.run()

    if Config.get('predicting.run') and Config.get('predicting.predictor') is not None:
        wrapper = PredictorWrapper()
        wrapper.predictor = ModuleImporter.load(Config.get('predicting.predictor'))
        wrapper.run()


if __name__ == '__main__':
    main()
