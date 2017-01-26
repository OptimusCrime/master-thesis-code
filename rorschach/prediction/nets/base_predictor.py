#!/usr/bin/env python
# -*- coding: utf-8 -*-


class BasePredictor:

    def __init__(self):
        self.training_set = None
        self.test_set = None

        self.training_images_transformed = None
        self.training_labels_transformed = None
        self.test_images_transformed = None
        self.test_labels_transformed = None

    def prepare(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass
