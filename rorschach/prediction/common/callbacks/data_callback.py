#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

from rorschach.prediction.common.callbacks import BaseCallback
from rorschach.utilities import Config


class DataCallback(BaseCallback):

    def __init__(self):
        super().__init__()

    def run(self):
        dump_file = Config.get_path('path.output', 'data.json', fragment=Config.get('uid'))

        with open(dump_file, 'w') as outfile:
            json.dump(self.data.all(), outfile)
