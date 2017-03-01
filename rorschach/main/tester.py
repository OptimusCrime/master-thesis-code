#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Avoid creating a new directory each time we run this script
# flake8: noqa: E402


from rorschach.common import DataStore # isort:skip
DataStore.CONTENT['no-create'] = True

import os

from rorschach.prediction import PredictorWrapper
from rorschach.utilities import Config, LoggerWrapper, ModuleImporter

class Tester:

    def __init__(self):
        pass

    @staticmethod
    def run():
        if Config.get('general.mode') == 'test':
            Tester.provide_uid()

        log = LoggerWrapper.load(__name__)
        log.info('Current uid is %s', Config.get('uid'))
        log.info('Storing output in %s', Config.get_path('path.output', '', fragment=Config.get('uid')))

        if Config.get('predicting.run') and Config.get('predicting.predictor') is not None:
            wrapper = PredictorWrapper()
            wrapper.predictor = ModuleImporter.load(Config.get('predicting.predictor'))
            wrapper.test()

    @staticmethod
    def provide_uid():
        uid = input('Enter uid: ')

        if uid is None or len(uid) == 0:
            raise Exception('No valid uid provided')

        if not os.path.exists(Config.get_path('path.output', 'data.json', fragment=uid)
            raise Exception('No valid uid provided')

        Config.set('uid', uid)
