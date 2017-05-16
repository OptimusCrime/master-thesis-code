# -*- coding: utf-8 -*-

import glob
import os

from rorschach.prediction.common.callbacks import BaseCallback
from rorschach.utilities import Config


class EpochIndicatorCallback(BaseCallback):

    def __init__(self):
        super().__init__()

    def run(self):
        self.clean()
        self.create()

    def clean(self):
        output_dir = Config.get_path('path.output', Config.get('uid'))
        output_files = glob.glob(output_dir + os.sep + 'epoch_*')

        if len(output_files) == 0:
            return

        for file in output_files:
            os.remove(file)

    def create(self):
        file_name = 'epoch_' + str(self.data.get('epoch') + 1)
        open(Config.get_path('path.output', file_name, fragment=Config.get('uid')), 'a').close()
