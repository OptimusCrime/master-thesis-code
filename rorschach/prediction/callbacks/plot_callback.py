#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from keras.callbacks import Callback

from rorschach.utilities import Filesystem


class PlotCallback(Callback):

    def __init__(self):
        super().__init__()

        self.epochs = None

        self.data = {
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
        self.data['loss'].append(logs.get('loss'))
        self.data['val_loss'].append(logs.get('val_loss'))
        self.data['acc'].append(logs.get('acc'))
        self.data['val_acc'].append(logs.get('val_acc'))

        self.update_graph()

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def update_graph(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(self.data['loss'], label="loss")
        ax.plot(self.data['val_loss'], label="val_loss")
        ax.plot(self.data['acc'], label="acc")
        ax.plot(self.data['val_acc'], label="val_acc")

        ax.set_title('accuracy')
        ax.set_ylabel('values')
        ax.set_xlabel('epochs')

        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', bottom='off', top='off')

        ax.set_xlim(1, self.epochs)
        ax.set_xticks(np.arange(1, self.epochs + 1))

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=5)

        fig.savefig(Filesystem.get_root_path('data/accuracy.png'))
