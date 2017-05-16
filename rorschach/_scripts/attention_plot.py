#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable  # NOQA

from rorschach.utilities import Config


class AttentionPlot:

    LABELS = None
    CLASSES_INDEX = 0
    CLASSES = None
    RAW_CLASSES = None

    def __init__(self):
        pass

    def fetch_labels(self):
        with open(Config.get_path('path.data', 'labels_lookup.json')) as data_file:
            AttentionPlot.LABELS = json.load(data_file)

    @staticmethod
    def classes_pretty(data):
        pretty = []
        for key, value in data.items():
            pretty.append(value)
        return pretty[0]

    def fetch_classes(self):
        post_file = Config.get_path('path.output', 'rearrange_post.json', fragment=Config.get('uid'))
        pre_file = Config.get_path('path.output', 'rearrange_pre.json', fragment=Config.get('uid'))

        if not os.path.exists(post_file) or not os.path.exists(pre_file):
            AttentionPlot.CLASSES = []
            return None

        AttentionPlot.RAW_CLASSES = {
            'post': None,
            'pre': None
        }

        with open(post_file) as data_file:
            AttentionPlot.RAW_CLASSES['post'] = AttentionPlot.classes_pretty(json.load(data_file))

        with open(pre_file) as data_file:
            AttentionPlot.RAW_CLASSES['pre'] = AttentionPlot.classes_pretty(json.load(data_file))

        AttentionPlot.CLASSES = dict()

    def find_class(self, value):
        found = False
        for i in range(AttentionPlot.CLASSES_INDEX, len(AttentionPlot.RAW_CLASSES['post'])):
            post = AttentionPlot.RAW_CLASSES['post'][i]
            pre = AttentionPlot.RAW_CLASSES['pre'][i]

            for j in range(min(len(post), len(pre))):
                if post[j] not in AttentionPlot.CLASSES:
                    AttentionPlot.CLASSES[post[j]] = pre[j]

                    if post[j] == value:
                        found = True

            if found:
                AttentionPlot.CLASSES_INDEX = i

                return AttentionPlot.CLASSES[value]

        raise Exception('Could not find sign ' + str(value))

    def lookup_class(self, value):
        if AttentionPlot.CLASSES is None:
            self.fetch_classes()

        if AttentionPlot.RAW_CLASSES is None:
            return value

        if value in AttentionPlot.CLASSES:
            return AttentionPlot.CLASSES[value]

        return self.find_class(value)

    def lookup_classes(self, classes):
        output_classes = []
        for _class in classes:
            output_classes.append(self.lookup_class(_class))
        return output_classes

    def lookup_labels(self, labels):
        if AttentionPlot.LABELS is None:
            self.fetch_labels()

        output_labels = []
        for label in labels:
            for key, value in AttentionPlot.LABELS.items():
                if value == label:
                    output_labels.append(key)

        return output_labels

    def plot(self, data, classes, labels):

        # Input values
        classes = classes.flatten()
        classes = self.lookup_classes(classes)

        classes_offset = 0
        zero_padds = 0
        for i in range(len(classes)):
            if classes[i] == 0:
                zero_padds += 1
                if zero_padds == 4:
                    classes_offset = i
                    break

        if classes_offset > 0:
            classes = classes[:classes_offset]

        # Output labels
        labels = labels.flatten()
        labels = self.lookup_labels(labels)

        data = np.array(data)

        if classes_offset == 0:
            data = data.reshape((data.shape[0], data.shape[-1]))[:len(labels), :]
        else:
            data = data.reshape((data.shape[0], data.shape[-1]))[:len(labels), :classes_offset]

        fig = plt.figure(figsize=(20, 20), dpi=80)

        ax = fig.add_subplot(111)
        ax.imshow(data,
                  interpolation='nearest',
                  cmap=plt.cm.Blues,
                  )

        ax.set_xticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)

        thresh = data.max() / 2.
        for i, j in itertools.product(range(data.shape[0]), range(data.shape[1])):
            ax.text(
                j,
                i,
                " ",
                horizontalalignment="center",
                color="white" if data[i, j] > thresh else "black"
            )

        plt.tight_layout()

        fig.savefig(Config.get_path('path.output', 'attention.png', fragment=Config.get('uid')))
