#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np

from rorschach.utilities import Config, unpickle_data


class ConfusionMatrix:

    INT = 0
    ARGMAX = 1

    def __init__(self):
        self.correct_format = None
        self.confusion_matrix = None
        self.null_value = None

    def create_matrix(self, data):
        self.confusion_matrix = np.zeros((len(data) - 1, len(data) - 1), dtype=np.int32)
        self.null_value = len(data) - 2

    def populate_matrix(self, data):
        # Foreach word
        for i in range(len(data['predictions'])):
            prediction = data['predictions'][i]
            correct = data['correct'][i]

            # Foreach letter
            for j in range(len(prediction)):
                self.handle_values(prediction[j], correct[j])

    def convert_values(self, value):
        if value == 0:
            return self.null_value

        return value - 1

    def handle_values(self, prediction, correct):
        prediction_value = np.argmax(prediction)
        correct_value = self.handle_correct(correct)

        prediction_value = self.convert_values(prediction_value)
        correct_value = self.convert_values(correct_value)

        self.confusion_matrix[prediction_value][correct_value] += 1

    def handle_correct(self, correct):
        if self.correct_format is None:
            if type(correct) is int:
                self.correct_format = ConfusionMatrix.INT
            else:
                self.correct_format = ConfusionMatrix.ARGMAX

        if self.correct_format == ConfusionMatrix.ARGMAX:
            return np.argmax(correct)

        return correct

    def run(self):
        data = unpickle_data(Config.get_path('path.output', 'predictions.pickl', fragment=Config.get('uid')))

        self.create_matrix(data['predictions'][0][0])
        self.populate_matrix(data)

        self.debug()

    def debug(self):
        for i in range(len(self.confusion_matrix)):
            print(' '.join(str(x) for x in self.confusion_matrix[i]))


if __name__ == '__main__':
    confusion_matrix = ConfusionMatrix()
    confusion_matrix.run()
