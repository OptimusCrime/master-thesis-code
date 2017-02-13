#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker

from rorschach.prediction.tensorflow.callbacks import CallbackRunner
from rorschach.utilities import Config


class CallbackPlotter():

    def __init__(self):
        super().__init__()

        self.data = {}
        self.callback_type = None

    def run(self):
        data = self.graph_data()

        # Do not produce any figure the first epoch, not enough data
        for key, value in data.items():
            if len(value) == 1:
                return

        fig, ax_loss, ax_acc = self.build_axes()

        self.add_plots(data, ax_loss, ax_acc)
        self.add_labels(ax_loss, ax_acc)
        self.set_ticks(data, ax_loss, ax_acc)
        self.adjust_legend(ax_loss, ax_acc)

        self.save_plot(fig)

    def graph_data(self):
        if self.callback_type == CallbackRunner.TRAINING:
            return {
                'loss': self.data['loss']
            }

        return {
            'test_loss': self.data['test_loss'],
            'test_accuracy': self.data['test_accuracy']
        }

    def build_axes(self):
        fig = plt.figure(figsize=(16, 6), dpi=80)

        ax_loss = None
        ax_acc = None

        if self.callback_type == CallbackRunner.TRAINING:
            ax_loss = fig.add_subplot(111)
        else:
            ax_loss = fig.add_subplot(121)
            ax_acc = fig.add_subplot(122)

        return fig, ax_loss, ax_acc

    def add_plots(self, data, loss, acc):
        if self.callback_type == CallbackRunner.TEST:
            loss.plot(data['test_loss'], label="test_loss")
            acc.plot(data['test_accuracy'], label="test_accuracy")

            return

        # Add plots
        loss.plot(data['loss'], label="loss")

    def add_labels(self, loss, acc):
        # Set labels and titles
        loss.set_title('loss')
        loss.set_ylabel('loss')

        if self.callback_type == CallbackRunner.TEST:
            acc.set_title('accuracy')
            acc.set_ylabel('accuracy')

    def set_ticks(self, data, loss, acc):
        loss.minorticks_on()
        loss.tick_params(axis='x', which='major', labeltop=False, labelright=False, top=False)
        loss.tick_params(axis='x', which='minor', labeltop=False, labelright=False, top=False, bottom=False)
        loss.tick_params(axis='y', which='both', labeltop=False, labelright=True, right=True)

        if self.callback_type == CallbackRunner.TRAINING:
            loss.set_xlim(xmin=0, xmax=len(data['loss']) - 1)
        else:
            loss.set_xlim(xmin=0, xmax=len(data['test_loss']) - 1)
            acc.set_xlim(xmin=0, xmax=len(data['test_accuracy']) - 1)

        loss.set_ylim(ymin=0)
        loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        if self.callback_type == CallbackRunner.TEST:
            acc.minorticks_on()
            acc.tick_params(axis='x', which='major', labeltop=False, labelright=False, top=False)
            acc.tick_params(axis='x', which='minor', labeltop=False, labelright=False, top=False, bottom=False)
            acc.tick_params(axis='y', which='both', labeltop=False, labelright=True, right=True)

            acc.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            # Static y max/min on accuracy
            acc.set_ylim(0., 1.)
            acc.set_yticks(np.arange(0., 1.1, 0.1))

    def adjust_legend(self, loss, acc):
        # Fix legend below the graph
        box_loss = loss.get_position()

        if self.callback_type == CallbackRunner.TRAINING:
            loss.set_position([box_loss.x0,
                               box_loss.y0 + box_loss.height * 0.12,
                               box_loss.width,
                               box_loss.height * 0.88])
        else:
            loss.set_position([box_loss.x0 - box_loss.width * 0.12,  # Move to the left
                              box_loss.y0 + box_loss.height * 0.12,
                              box_loss.width,
                              box_loss.height * 0.88])

            box_acc = acc.get_position()
            acc.set_position([box_acc.x0 + box_acc.width * 0.1,  # Move to the right
                              box_acc.y0 + box_acc.height * 0.12,
                              box_acc.width,
                              box_acc.height * 0.88])

            acc.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
                       fancybox=True, shadow=True, ncol=5)

        loss.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
                    fancybox=True, shadow=True, ncol=5)

    def save_plot(self, fig):
        file_name = 'plot_test'
        if self.callback_type == CallbackRunner.TRAINING:
            file_name = 'plot_training'

        fig.savefig(Config.get_path('path.output', file_name + '.png', fragment=Config.get('uid')))
        plt.close()
