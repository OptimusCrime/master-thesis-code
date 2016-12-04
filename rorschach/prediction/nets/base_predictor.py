#!/usr/bin/env python
# -*- coding: utf-8 -*-


class BasePredictor:

    def __init__(self):
        self.data_set = None
        self.phrase = None
        self.predictions = []

        self.training_images_transformed = None
        self.training_labels_transformed = None
        self.predicting_image_transformed = None
        self.predicting_label_transformed = None

    def preprocess(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass
