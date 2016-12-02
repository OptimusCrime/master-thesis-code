#!/usr/bin/env python
# -*- coding: utf-8 -*-


class BasePredictor:

    def __init__(self):
        self.data_set = None
        self.phrase = None
        self.predictions = []

    def preprocess(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass
