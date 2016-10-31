#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utilities import Filesystem, unpickle_data


class PredictorWrapper:

    def __init__(self):
        pass

    def run(self):
        print(unpickle_data(Filesystem.get_root_path('data/phrase.pickl')))
        pass

    @property
    def predictions(self):
        return []
