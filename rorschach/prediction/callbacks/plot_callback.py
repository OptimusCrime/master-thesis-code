#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
from keras.callbacks import Callback

from rorschach.utilities import Filesystem


class PlotCallback(Callback):

    EPOCH = 0
    BATCH = 1

    def __init__(self):
        super().__init__()

        self.epochs = None

        self.data_epoch = {
            'loss': [0],
            'val_loss': [0],
            'acc': [0],
            'val_acc': [0]
        }

        self.data_batch = {
            'loss': [],
            'val_loss': [],
            'acc': [],
            'val_acc': []
        }

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.data_epoch['loss'].append(logs.get('loss'))
        self.data_epoch['val_loss'].append(logs.get('val_loss'))
        self.data_epoch['acc'].append(logs.get('acc'))
        self.data_epoch['val_acc'].append(logs.get('val_acc'))

        self.update_graph(PlotCallback.EPOCH)

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        self.data_batch['loss'].append(logs.get('loss'))
        self.data_batch['val_loss'].append(logs.get('val_loss'))
        self.data_batch['acc'].append(logs.get('acc'))
        self.data_batch['val_acc'].append(logs.get('val_acc'))

        self.update_graph(PlotCallback.BATCH)

    def update_graph(self, plot_type):
        data = self.graph_data(plot_type)

        fig, ax_loss, ax_acc = self.build_axes()

        PlotCallback.add_plots(data, ax_loss, ax_acc, plot_type)
        PlotCallback.add_labels(ax_loss, ax_acc, plot_type)
        self.set_ticks(data, ax_loss, ax_acc, plot_type)
        PlotCallback.adjust_legend(ax_loss, ax_acc)

        self.save_plot(fig, plot_type)

    def graph_data(self, plot_type):
        if plot_type == PlotCallback.EPOCH:
            return self.data_epoch

        return self.data_batch

    @staticmethod
    def build_axes():
        fig = plt.figure(figsize=(16, 6), dpi=80)

        # Subplots
        ax_loss = fig.add_subplot(121)
        ax_acc = fig.add_subplot(122)

        return fig, ax_loss, ax_acc

    @staticmethod
    def add_plots(data, loss, acc, plot_type):
        # Add plots
        loss.plot(data['loss'], label="loss")

        if plot_type == PlotCallback.EPOCH:
            loss.plot(data['val_loss'], label="val_loss")

        acc.plot(data['acc'], label="acc")

        if plot_type == PlotCallback.EPOCH:
            acc.plot(data['val_acc'], label="val_acc")

    @staticmethod
    def add_labels(loss, acc, plot_type):
        # Set labels and titles
        loss.set_title('loss')
        loss.set_ylabel('loss')
        loss.set_xlabel('epochs')

        acc.set_title('accuracy')
        acc.set_ylabel('accuracy')
        acc.set_xlabel('epochs')

        if plot_type == PlotCallback.BATCH:
            loss.set_xlabel('batch')
            acc.set_xlabel('batch')

    def set_ticks(self, data, loss, acc, plot_type):
        loss.minorticks_on()
        loss.tick_params(axis='x', which='major', labeltop=False, labelright=False, top=False)
        loss.tick_params(axis='x', which='minor', labeltop=False, labelright=False, top=False, bottom=False)
        loss.tick_params(axis='y', which='both', labeltop=False, labelright=True, right=True)

        acc.minorticks_on()
        acc.tick_params(axis='x', which='major', labeltop=False, labelright=False, top=False)
        acc.tick_params(axis='x', which='minor', labeltop=False, labelright=False, top=False, bottom=False)
        acc.tick_params(axis='y', which='both', labeltop=False, labelright=True, right=True)

        # Set x limit and ticks
        loss.set_xlim(xmin=0, xmax=len(data['loss']) - 1)
        acc.set_xlim(xmin=0, xmax=len(data['loss']) - 1)

        loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        acc.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        if plot_type == PlotCallback.EPOCH:
            loss.set_xlim(1, self.epochs)
            loss.set_xticks(np.arange(1, self.epochs + 1))

            acc.set_xlim(1, self.epochs)
            acc.set_xticks(np.arange(1, self.epochs + 1))

        # Static y max/min on accuracy
        acc.set_ylim(0., 1.)
        acc.set_yticks(np.arange(0., 1.1, 0.1))

    @staticmethod
    def adjust_legend(loss, acc):
        # Fix legend below the graph
        box_loss = loss.get_position()
        loss.set_position([box_loss.x0 - box_loss.width * 0.12,  # Move to the left
                           box_loss.y0 + box_loss.height * 0.12,
                           box_loss.width,
                           box_loss.height * 0.88])

        box_acc = acc.get_position()
        acc.set_position([box_acc.x0 + box_acc.width * 0.1,  # Move to the right
                          box_acc.y0 + box_acc.height * 0.12,
                          box_acc.width,
                          box_acc.height * 0.88])

        loss.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
                    fancybox=True, shadow=True, ncol=5)

        acc.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
                   fancybox=True, shadow=True, ncol=5)

    @staticmethod
    def save_plot(fig, plot_type):
        file_name = 'plot_epoch'
        if plot_type == PlotCallback.BATCH:
            file_name = 'plot_batch'

        fig.savefig(Filesystem.get_root_path('data/' + file_name + '.png'))
        plt.close()
