#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.utilities import Config, Filesystem, unpickle_data


class MatrixDim:

    UNIQUE_SIZE = None

    def __init__(self):
        pass

    @staticmethod
    def get_size():
        if Config.get('preprocessing.unique-signatures'):
            return MatrixDim.get_unique_size()

        return len(Config.get('characters'))

    @staticmethod
    def get_unique_size():
        if MatrixDim.UNIQUE_SIZE is None:
            MatrixDim.UNIQUE_SIZE = unpickle_data(Filesystem.get_root_path('data/unique_signatures.pickl'))

        return MatrixDim.UNIQUE_SIZE
