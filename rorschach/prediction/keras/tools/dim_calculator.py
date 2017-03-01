#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.utilities import Config, unpickle_data


class DimCalculator:

    def __init__(self):
        pass

    @staticmethod
    def width(data_set):
        return data_set[0].shape[-1]

    @staticmethod
    def depth():
        return len(unpickle_data(Config.get_path('path.data', 'labels.pickl')))
