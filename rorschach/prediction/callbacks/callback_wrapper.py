#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
from keras.callbacks import Callback

from rorschach.utilities import Filesystem


class CallbackWrapper(Callback):

    def __init__(self, callbacks):
        super().__init__()

        self.callbacks = callbacks
        self.epochs = None

    def on_train_begin(self, logs={}):
        self.execute('on_train_begin', logs)

    def on_train_end(self, logs={}):
        self.execute('on_train_end', logs)

    def on_epoch_begin(self, epoch, logs={}):
        self.execute('on_epoch_begin', epoch, logs)

    def on_epoch_end(self, epoch, logs={}):
        self.execute('on_epoch_end', epoch, logs)

    def on_batch_begin(self, batch, logs={}):
        self.execute('on_batch_begin', batch, logs)

    def on_batch_end(self, batch, logs={}):
        self.execute('on_batch_end', batch, logs)

    def execute(self, method, *args):
        for callback in self.callbacks:
            callback_thread = callback()
            callback_thread.start()
            dynamic_call = getattr(callback_thread, method)
            dynamic_call(args)

