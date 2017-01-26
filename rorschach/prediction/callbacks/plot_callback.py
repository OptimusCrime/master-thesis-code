#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
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
            'loss': [0],
            'val_loss': [0],
            'acc': [0],
            'val_acc': [0]
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

    def update_graph(self, type):
        data = self.data_epoch

        if type == PlotCallback.BATCH:
            data = self.data_batch

        fig = plt.figure(figsize=(16, 6), dpi=80)

        # Subplots
        ax_loss = fig.add_subplot(121)
        ax_acc = fig.add_subplot(122)

        # Add plots
        ax_loss.plot(data['loss'], label="loss")
        ax_loss.plot(data['val_loss'], label="val_loss")

        ax_acc.plot(data['acc'], label="acc")
        ax_acc.plot(data['val_acc'], label="val_acc")

        # Set labels and titles
        ax_loss.set_title('loss')
        ax_loss.set_ylabel('loss')
        ax_loss.set_xlabel('epochs')

        ax_acc.set_title('accuracy')
        ax_acc.set_ylabel('accuracy')
        ax_acc.set_xlabel('epochs')

        if type == PlotCallback.BATCH:
            ax_loss.set_xlabel('batch')
            ax_acc.set_xlabel('batch')

        # Ticks
        ax_loss.minorticks_on()
        ax_loss.tick_params(labeltop=False, labelright=True)

        ax_acc.minorticks_on()
        ax_acc.tick_params(labeltop=False, labelright=True)

        # Set x limit and ticks
        if type == PlotCallback.EPOCH:
            ax_loss.set_xlim(1, self.epochs)
            ax_loss.set_xticks(np.arange(1, self.epochs + 1))

            ax_acc.set_xlim(1, self.epochs)
            ax_acc.set_xticks(np.arange(1, self.epochs + 1))

        # Static y max/min on accuracy
        ax_acc.set_ylim(0., 1.)
        ax_acc.set_yticks(np.arange(0., 1.1, 0.1))

        # Fix legend below the graph
        box_loss = ax_loss.get_position()
        ax_loss.set_position([box_loss.x0 - box_loss.width * 0.12,  # Move to the left
                              box_loss.y0 + box_loss.height * 0.12,
                              box_loss.width,
                              box_loss.height * 0.88])

        box_acc = ax_acc.get_position()
        ax_acc.set_position([box_acc.x0 + box_acc.width * 0.1,  # Move to the right
                             box_acc.y0 + box_acc.height * 0.12,
                             box_acc.width,
                             box_acc.height * 0.88])

        ax_loss.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
                       fancybox=True, shadow=True, ncol=5)

        ax_acc.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
                      fancybox=True, shadow=True, ncol=5)

        file_name = 'plot_epoch'
        if type == PlotCallback.BATCH:
            file_name = 'plot_batch'

        fig.savefig(Filesystem.get_root_path('data/' + file_name + '.png'))
