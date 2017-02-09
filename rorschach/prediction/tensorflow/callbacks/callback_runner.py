#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.utilities import Config


class CallbackRunner():

    TRAINING = 0
    TEST = 1

    def __init__(self, callbacks):
        super().__init__()

        self.callbacks = callbacks

        self.data = {
            'epochs': Config.get('predicting.epochs'),
        }

    def run(self, values, callback_type):
        self.add_values(values)
        self.run_callbacks(callback_type)

    def add_values(self, values):
        for key, value in values.items():
            if key not in self.data:
                self.data[key] = [value]
                continue

            self.data[key].append(value)

    def run_callbacks(self, callback_type):
        for callback in self.callbacks:
            thread = callback()
            thread.data = self.data
            thread.callback_type = callback_type
            thread.start()
