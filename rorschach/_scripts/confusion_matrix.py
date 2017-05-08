#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from rorschach.utilities import Config, unpickle_data


class ConfusionMatrix:

    INT = 0
    ARGMAX = 1

    def __init__(self):
        self.correct_format = None
        self.confusion_matrix = None
        self.null_value = None

    def create_matrix(self, data, list_type):
        offset = 0
        if list_type == False:
            offset = 1

        self.confusion_matrix = np.zeros((len(data) - offset, len(data) - offset), dtype=np.int32)
        self.null_value = self.confusion_matrix.shape[0] - 1

    def populate_matrix(self, data):
        # Foreach word
        for i in range(len(data['correct'])):
            prediction = data['predictions'][i]
            correct = data['correct'][i]

            # Foreach letter
            for j in range(len(correct)):
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

        self.confusion_matrix[correct_value][prediction_value] += 1

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

        self.create_matrix(data['predictions'][0][0], type(data['correct'][0] == list))
        self.populate_matrix(data)

        self.plot()

        self.debug()

    def plot(self):
        classes = Config.get('general.characters') + ['-']
        normalize = True

        if normalize:
            self.confusion_matrix = self.confusion_matrix.astype('float') / self.confusion_matrix.sum(axis=1)[:, np.newaxis]

            for i in range(len(self.confusion_matrix)):
                for j in range(len(self.confusion_matrix[0])):
                    if np.isnan(self.confusion_matrix[i][j]):
                        self.confusion_matrix[i][j] = 0.0

        size = 20
        if len(classes) > 30:
            size = 30

        fig = plt.figure(figsize=(size, size), dpi=80)

        ax = fig.add_subplot(111)

        im = ax.imshow(self.confusion_matrix,
                   interpolation='nearest',
                   cmap=plt.cm.Blues,
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.4)
        fig.colorbar(im, cax=cax)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)

        thresh = self.confusion_matrix.max() / 2.
        for i, j in itertools.product(range(self.confusion_matrix.shape[0]), range(self.confusion_matrix.shape[1])):
            ax.text(j, i, "{0:0.3f}".format(self.confusion_matrix[i, j]),
                     horizontalalignment="center",
                     color="white" if self.confusion_matrix[i, j] > thresh else "black")

        plt.tight_layout()

        fig.savefig(Config.get_path('path.output', 'confusion_matrix.png', fragment=Config.get('uid')))

    def debug(self):
        for i in range(len(self.confusion_matrix)):
            print(' '.join(str(x) for x in self.confusion_matrix[i]))


if __name__ == '__main__':
    confusion_matrix = ConfusionMatrix()
    confusion_matrix.run()
