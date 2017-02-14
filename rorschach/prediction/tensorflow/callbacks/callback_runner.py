#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.utilities import Config


class CallbackRunner():

    LOSS = 0
    ACCURACY = 1

    def __init__(self, callbacks):
        super().__init__()

        self.callbacks = callbacks

    def run(self, values, callback_type):
        for callback in self.callbacks:
            instance = callback()
            instance.data = values
            instance.callback_type = callback_type
            instance.run()
