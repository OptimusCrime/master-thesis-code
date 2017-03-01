#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
from matplotlib.patches import Ellipse

from rorschach.prediction.common.callbacks import BaseCallback
from rorschach.utilities import Config


class PlotterCallback(BaseCallback):

    LOSS = 0
    ACCURACY = 1

    CIRCLE_WIDTH = 0.15
    FIGURE_WIDTH = 16
    FIGURE_HEIGHT = 6

    def __init__(self):
        super().__init__()

    def run(self):
        # Do not produce any figure the first epoch, not enough data
        for key, value in self.data.all().items():
            if key != 'stores' and len(value) == 1:
                return

        fig, ax = self.build_axes()

        self.add_plots(ax)
        self.add_labels(ax)
        self.set_ticks(ax)
        self.adjust_legend(ax)

        if 'stores' in self.data and len(self.data['stores']) > 0:
            self.add_stores(ax)

        self.save_plot(fig)

    def build_axes(self):
        fig = plt.figure(figsize=(PlotterCallback.FIGURE_WIDTH, PlotterCallback.FIGURE_HEIGHT), dpi=80)

        ax = fig.add_subplot(111)

        return fig, ax

    def add_plots(self, ax):
        if PlotterCallback.LOSS in self.flags:
            ax.plot(self.data['loss_train'], label="training")
            ax.plot(self.data['loss_validate'], label="validation")

            return

        # Add plots
        ax.plot(self.data['accuracy'], label="accuracy")

    def add_labels(self, ax):
        if PlotterCallback.LOSS in self.flags:
            ax.set_title('loss')
            ax.set_ylabel('loss')

            return

        ax.set_title('accuracy')
        ax.set_ylabel('accuracy')

    def set_ticks(self, ax):
        ax.minorticks_on()
        ax.tick_params(axis='x', which='major', labeltop=False, labelright=False, top=False)
        ax.tick_params(axis='x', which='minor', labeltop=False, labelright=False, top=False, bottom=False)
        ax.tick_params(axis='y', which='both', labeltop=False, labelright=True, right=True)
        ax.set_ylim(ymin=0)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Specific for loss
        if PlotterCallback.LOSS in self.flags:
            ax.set_xlim(xmin=0, xmax=len(self.data['loss_validate']) - 1)

            return

        # Specific for accuracy
        ax.set_xlim(xmin=0, xmax=len(self.data['accuracy']) - 1)
        ax.set_ylim(0., 1.)
        ax.set_yticks(np.arange(0., 1.1, 0.1))

    def adjust_legend(self, ax):
        # Fix legend below the graph
        box_loss = ax.get_position()

        ax.set_position([box_loss.x0,
                         box_loss.y0 + box_loss.height * 0.12,
                         box_loss.width * 0.88,
                         box_loss.height * 0.88])

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
                  fancybox=True, shadow=True, ncol=5)

    def add_stores(self, ax):
        for epoch in self.data['stores']:
            value = self.data['loss_validate'][epoch]

            # Because axis are different
            ellipse_width = PlotterCallback.CIRCLE_WIDTH * (
                PlotterCallback.FIGURE_HEIGHT / PlotterCallback.FIGURE_WIDTH) * 2
            ellipse_height = PlotterCallback.CIRCLE_WIDTH

            store = Ellipse((epoch, value), ellipse_width, ellipse_height, color='r', alpha=0.3)
            ax.add_artist(store)

    def save_plot(self, fig):
        file_name = 'plot_loss'
        if PlotterCallback.ACCURACY in self.flags:
            file_name = 'plot_accuracy'

        fig.savefig(Config.get_path('path.output', file_name + '.png', fragment=Config.get('uid')))
        plt.close()