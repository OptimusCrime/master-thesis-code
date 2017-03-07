#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.callbacks import Callback

from rorschach.prediction.common.callbacks import DataCallback, KerasSaverCallback, PlotterCallback


class KerasCallbackRunnerBridge(Callback):

    def __init__(self, callback_runner):
        super().__init__()

        self.model = None
        self.runner = callback_runner

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        # Update information in our data container
        self.runner.data_container.set('epoch', epoch)

        if 'val_acc' in logs:
            self.runner.data_container.add_list('validate_accuracy', logs['val_acc'])

        if 'val_loss' in logs:
            self.runner.data_container.add_list('validate_loss', logs['val_loss'])

        if 'loss' in logs:
            self.runner.data_container.add_list('train_loss', logs['loss'])

        # Other keras log values
        if 'val_categorical_accuracy' in logs:
            self.runner.data_container.add_list('validate_accuracy', logs['val_categorical_accuracy'])

        # Run all the callbacks
        self.runner.run([KerasSaverCallback], None, {
            'model': self.model
        })
        self.runner.run([PlotterCallback], [PlotterCallback.LOSS])
        self.runner.run([PlotterCallback], [PlotterCallback.ACCURACY])
        self.runner.run([DataCallback])

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass
