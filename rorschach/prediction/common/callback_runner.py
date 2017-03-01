#!/usr/bin/env python
# -*- coding: utf-8 -*-


class CallbackRunner():

    def __init__(self, data_container):
        self.data_container = data_container

    def run(self, callbacks, flags=None, information=None):
        for callback in callbacks:
            instance = callback()
            instance.data = self.data_container
            instance.flags = flags
            instance.information = information
            instance.run()
