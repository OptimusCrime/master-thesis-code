#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Avoid creating a new directory each time we run this script
# flake8: noqa: E402


from rorschach.common import DataStore # isort:skip
DataStore.CONTENT['no-create'] = True

import json
import os

from rorschach.prediction.tensorflow.callbacks import CallbackRunner
from rorschach.prediction.tensorflow.callbacks.plotter import CallbackPlotter
from rorschach.utilities import Config


class ReplotData:

    def __init__(self):
        # Force load Config to avoid overwriting the uid
        Config.load_config()

        self.uid = None
        self.callback = CallbackRunner([
            CallbackPlotter
        ])

    def run(self):
        self.find_uid()

        content = self.load_json()
        if content is None:
            raise Exception('No valid data found in json file')

        self.replot(content)

    def find_uid(self):
        self.uid = input('Enter uid for replotting: ')

        if self.uid is None or len(self.uid) == 0:
            raise Exception('No valid uid provided')

        # Store in Config (because the plotter uses it)
        Config.set('uid', self.uid)

    def load_json(self):
        data_path = Config.get_path('path.output', 'data.json', fragment=self.uid)

        if not os.path.exists(data_path):
            raise Exception('Could not find file in ' + data_path)

        with open(data_path) as data_file:
            data = json.load(data_file)

            if type(data) == dict:
                return data

        return None

    def replot(self, content):
        self.callback.run({
            'loss_train': content['train_loss'],
            'loss_validate': content['validate_loss'],
            'stores': content['stores']
        }, CallbackRunner.LOSS)

        # Plot the accuracy
        self.callback.run({
            'accuracy': content['validate_accuracy']
        }, CallbackRunner.ACCURACY)

        print('Successfully replotted data')


if __name__ == '__main__':
    replot_data = ReplotData()
    replot_data.run()
