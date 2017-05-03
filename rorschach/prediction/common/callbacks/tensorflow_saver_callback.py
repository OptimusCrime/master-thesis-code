# -*- coding: utf-8 -*-

import os
import re

from rorschach.prediction.common.callbacks import BaseSaverCallback
from rorschach.utilities import Config


class TensorflowSaverCallback(BaseSaverCallback):

    MODEL_CKPT_PATTERN = re.compile('^model\.ckpt-[0-9]\.(?:index|meta|data\-[0-9]*\-of\-[0-9]*)$')

    def __init__(self):
        super().__init__()

    def run(self):
        if self.should_save_session():
            self.information['log'].write('- Saving model')

            self.save_session(self.information['saver'], self.data.get('epoch'))

    def save_session(self, saver, epoch):
        # Add save to plot
        self.data.add_list('stores', epoch)

        should_save = Config.get('various.save-indicator')
        save_indicator = Config.get_path('path.output', 'save', fragment=Config.get('uid'))
        if should_save in [False, None] or (should_save and os.path.exists(save_indicator)):
            # Save the ckpt dump
            saver.save(
                self.information['session'],
                Config.get_path('path.output', 'model.ckpt', fragment=Config.get('uid')),
                global_step=epoch
            )
