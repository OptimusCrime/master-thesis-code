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
