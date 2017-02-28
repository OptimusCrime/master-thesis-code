#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

from rorschach.utilities import Config


class DataDumper:

    def __init__(self):
        pass

    @staticmethod
    def dump(data):
        dump_file = Config.get_path('path.output', 'data.json', fragment=Config.get('uid'))

        with open(dump_file, 'w') as outfile:
            json.dump(data, outfile)
