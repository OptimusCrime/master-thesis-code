#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import re

from keras.callbacks import Callback


class CallbackWrapper(Callback):

    EPOCH = 'epoch'
    BATCH = 'batch'

    def __init__(self, callbacks):
        super().__init__()

        self.callbacks = callbacks

        self.data = {
            'epochs': None,
            CallbackWrapper.EPOCH: {
                'loss': [0],
                'val_loss': [0],
                'acc': [0],
                'val_acc': [0]
            },
            CallbackWrapper.BATCH: {
                'loss': [],
                'val_loss': [],
                'acc': [],
                'val_acc': []
            }
        }

    def on_train_begin(self, logs={}):
        self.execute('on_train_begin', logs)

    def on_train_end(self, logs={}):
        self.execute('on_train_end', logs)

    def on_epoch_begin(self, epoch, logs={}):
        self.execute('on_epoch_begin', epoch, logs)

    def on_epoch_end(self, epoch, logs={}):
        logs = CallbackWrapper.clean_logs(copy.copy(logs))

        self.data[CallbackWrapper.EPOCH]['loss'].append(logs.get('loss'))
        self.data[CallbackWrapper.EPOCH]['val_loss'].append(logs.get('val_loss'))
        self.data[CallbackWrapper.EPOCH]['acc'].append(logs.get('acc'))
        self.data[CallbackWrapper.EPOCH]['val_acc'].append(logs.get('val_acc'))

        self.execute('on_epoch_end', epoch, logs)

    def on_batch_begin(self, batch, logs={}):
        self.execute('on_batch_begin', batch, logs)

    def on_batch_end(self, batch, logs={}):
        logs = CallbackWrapper.clean_logs(copy.copy(logs))

        self.data[CallbackWrapper.BATCH]['loss'].append(logs.get('loss'))
        self.data[CallbackWrapper.BATCH]['acc'].append(logs.get('acc'))

        self.execute('on_batch_end', batch, logs)

    def execute(self, method, *args):
        for callback in self.callbacks:
            callback_thread = callback()

            callback_thread.data = self.data
            callback_thread.start()

            dynamic_call = getattr(callback_thread, method)
            dynamic_call(*args)

    REGEX_ACC_PATTERN = re.compile('^activation_(?:[0-9]*)_acc')
    REGEX_VAL_ACC_PATTERN = re.compile('^val_activation_(?:[0-9]*)_acc')

    @staticmethod
    def clean_logs(logs):
        new_logs = dict()
        if 'loss' in logs:
            new_logs['loss'] = logs['loss']

        if 'val_loss' in logs:
            new_logs['val_loss'] = logs['val_loss']

        average_acc = []
        average_val_acc = []
        for key, value in logs.items():
            if CallbackWrapper.REGEX_ACC_PATTERN.match(key):
                average_acc.append(value)

            if CallbackWrapper.REGEX_VAL_ACC_PATTERN.match(key):
                average_val_acc.append(value)

        if len(average_acc) > 0:
            new_logs['acc'] = sum(average_acc) / float(len(average_acc))

        if len(average_val_acc) > 0:
            new_logs['val_acc'] = sum(average_val_acc) / float(len(average_val_acc))

        return new_logs
