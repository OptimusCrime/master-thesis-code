# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker

from rorschach.prediction.common.callbacks import BaseCallback
from rorschach.utilities import Config


class PlotterCallback(BaseCallback):

    LOSS = 0
    ACCURACY = 1

    def __init__(self):
        super().__init__()

    def run(self):
        # Do not produce any figure the first epoch, not enough data
        for key in ['train_loss', 'validate_loss', 'validate_accuracy']:
            if self.data.has(key) and len(self.data.get(key)) == 1:
                return

        fig, ax = self.build_axes()

        if Config.get('plotter_max_epochs') is not None:
            self.adjust_plot()

        self.add_plots(ax)
        self.add_labels(ax)
        self.set_ticks(ax)
        self.adjust_legend(ax)

        if PlotterCallback.LOSS in self.flags and self.data.has('stores') and len(self.data.get('stores')) > 0:
            self.add_stores(ax)

        self.save_plot(fig)

    def build_axes(self):
        fig = plt.figure(figsize=(16, 6), dpi=80)

        ax = fig.add_subplot(111)

        if PlotterCallback.LOSS in self.flags:
            ax.grid(True, which="both")
        else:
            ax.grid(True)

        return fig, ax

    def adjust_plot(self):
        max_length = Config.get('plotter_max_epochs')
        self.data.set('train_loss', self.data.get('train_loss')[:max_length])
        self.data.set('validate_loss', self.data.get('validate_loss')[:max_length])
        self.data.set('validate_accuracy', self.data.get('validate_accuracy')[:max_length])

        new_stores = []
        for store in self.data.get('stores'):
            if store < max_length:
                new_stores.append(store)
        self.data.set('stores', new_stores)

    def add_plots(self, ax):
        if PlotterCallback.LOSS in self.flags:
            ax.plot(self.data.get('train_loss'), label="training")
            ax.plot(self.data.get('validate_loss'), label="validation")

            return

        # Add plots
        ax.plot(self.data.get('validate_accuracy'), label="accuracy")

    def add_labels(self, ax):
        ax.set_xlabel('epochs', fontsize=18)

        if PlotterCallback.LOSS in self.flags:
            ax.set_ylabel('loss', fontsize=18)
            return

        ax.set_ylabel('accuracy', fontsize=18)

    def set_ticks(self, ax):
        ax.minorticks_on()
        ax.tick_params(axis='x', which='major', labeltop=False, labelright=False, top=False)
        ax.tick_params(axis='x', which='minor', labeltop=False, labelright=False, top=False, bottom=False)
        ax.tick_params(axis='y', which='both', labeltop=False, labelright=True, right=True)

        if PlotterCallback.LOSS not in self.flags:
            ax.set_ylim(ymin=0)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Specific for loss
        if PlotterCallback.LOSS in self.flags:
            ax.set_xlim(xmin=0, xmax=len(self.data.get('validate_loss')) - 1)
            ax.set_yscale('log')

            return

        # Specific for accuracy
        ax.set_xlim(xmin=0, xmax=len(self.data.get('validate_accuracy')) - 1)
        ax.set_ylim(0., 1.)
        ax.set_yticks(np.arange(0., 1.1, 0.1))

    def adjust_legend(self, ax):
        if PlotterCallback.LOSS in self.flags:
            legend = ax.legend(loc='upper right')

            legend.get_frame().set_alpha(1)

        return None

        # Fix legend below the graph
        box_loss = ax.get_position()

        ax.set_position([box_loss.x0,
                         box_loss.y0 + box_loss.height * 0.12,
                         box_loss.width * 0.88,
                         box_loss.height * 0.88])

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
                  fancybox=True, shadow=True, ncol=5)

    def add_stores(self, ax):
        values = []
        sizes = []
        for epoch in self.data.get('stores'):
            values.append(self.data.get('validate_loss')[epoch])
            sizes.append(100)

        ax.scatter(self.data.get('stores'), values, s=sizes, color='red', alpha=0.3)

    def save_plot(self, fig):
        fig.tight_layout()

        file_name = 'plot_loss'
        if PlotterCallback.ACCURACY in self.flags:
            file_name = 'plot_accuracy'

        fig.savefig(Config.get_path('path.output', file_name + '.png', fragment=Config.get('uid')))
        plt.close()
