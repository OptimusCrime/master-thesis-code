#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta


class AbstractPredictor:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.data_set = None
        self.phrase = None
        self.predictions = []

    def preprocess(self):
        pass

    def predict(self):
        pass
