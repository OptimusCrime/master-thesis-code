#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.prediction.common import DataContainer


class BasePredictor:

    def __init__(self):
        self.data = {}

        self.data_container = DataContainer()

        self.training_set = None
        self.test_set = None
        self.validate_set = None

        self.training_images_transformed = None
        self.training_labels_transformed = None
        self.validate_images_transformed = None
        self.validate_labels_transformed = None
        self.test_images_transformed = None
        self.test_labels_transformed = None

    def prepare(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
