#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

from rorschach.prediction.common import CallbackRunner, DataContainer
from rorschach.prediction.common.callbacks import PlotterCallback
from rorschach.utilities import Config


class ReplotData:

    def __init__(self):
        # Force load Config to avoid overwriting the uid
        Config.load_config()

        self.uid = None
        self.data_container = DataContainer()
        self.callback = CallbackRunner(self.data_container)

    def run(self):
        self.find_uid()
        self.find_max_epochs()

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

    def find_max_epochs(self):
        max_epochs = input('Enter maximum number of epochs to plot: ')
        if max_epochs is None or len(max_epochs) == 0 or not str(max_epochs).isdigit():
            return None
        Config.set('plotter_max_epochs', int(max_epochs))

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
        self.data_container.reset(content)

        self.callback.run([PlotterCallback], [PlotterCallback.LOSS])
        self.callback.run([PlotterCallback], [PlotterCallback.ACCURACY])

        print('Successfully replotted data')


if __name__ == '__main__':
    replot_data = ReplotData()
    replot_data.run()
