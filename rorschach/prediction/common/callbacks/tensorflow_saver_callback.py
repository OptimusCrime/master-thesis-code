#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import os
import re

from rorschach.prediction.common.callbacks import BaseCallback
from rorschach.utilities import Config


class TensorflowSaverCallback(BaseCallback):

    MODEL_CKPT_PATTERN = re.compile('^model\.ckpt-[0-9]\.(?:index|meta|data\-[0-9]*\-of\-[0-9]*)')

    def __init__(self):
        super().__init__()

    def run(self):
        if self.should_save_session():
            self.information['log'].write('- Saving model')

            self.save_session(self.information['saver'], self.data.get('epoch'))

    def should_save_session(self):
        validation_loss = copy.copy(self.data.get('validate_loss'))
        if len(validation_loss) <= Config.get('predicting.best-results'):
            # We do not have enough data to store correctly. To make sure we always have some weights stored, do
            # following checks:

            # 1. If this is the first epoch dump weights either way
            if len(validation_loss) == 1:
                return True

            # 2. If we have more than one epochs, check if the current loss is "equal" to the overall best loss we have
            # seen this far
            current_loss = validation_loss[-1]
            validation_loss.pop(-1)

            # We are really just comparing floats here
            return abs(min(validation_loss) - current_loss) <= 0.001

        # The current (last) loss
        current_loss = validation_loss[-1]

        # The other loss values (without the last)
        validation_loss.pop(-1)

        # If the last value is higher or equal to the highest value in the other losses, we store the weights
        return current_loss <= min(validation_loss)

    def save_session(self, saver, epoch):
        # Add save to plot
        self.data.add_list('stores', epoch)

        # Delete earlier model files
        dir_content = os.listdir(os.path.join(Config.get('path.output'), Config.get('uid')))
        for content in dir_content:
            if not TensorflowSaverCallback.MODEL_CKPT_PATTERN.match(content):
                continue

            os.remove(Config.get_path('path.output', content, fragment=Config.get('uid')))

        # Save the ckpt dump
        saver.save(
            self.information['session'],
            Config.get_path('path.output', 'model.ckpt', fragment=Config.get('uid')),
            global_step=epoch
        )
