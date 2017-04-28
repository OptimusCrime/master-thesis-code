# -*- coding: utf-8 -*-

from rorschach.prediction import PredictorWrapper
from rorschach.preprocessing import Preprocessor
from rorschach.utilities import Config, LoggerWrapper, ModuleImporter, UidGenerator, UidGetter


class Trainer:

    def __init__(self):
        pass

    @staticmethod
    def run():
        if Config.get('general.mode') == 'continue':
            UidGetter.run()
        else:
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
            wrapper.train()
