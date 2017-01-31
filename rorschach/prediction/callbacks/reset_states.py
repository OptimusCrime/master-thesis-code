#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
from keras.callbacks import Callback

from rorschach.utilities import Filesystem


class ResetStates(Callback):

    EPOCH = 0
    BATCH = 1

    def __init__(self):
        super().__init__()

        self.counter = 0
        self.max_len = 20

    def on_batch_begin(self, batch, logs={}):
        self.model.reset_states()
        print("Resetting")
