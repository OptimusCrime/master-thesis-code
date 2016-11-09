#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import os


class Filesystem:

    def __init__(self):
        pass

    @staticmethod
    def get_source_path():
        # This line is stolen from stackoverflow.com/questions/50499/
        # Current directory of THIS file (utilities)
        utilities_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        # Go one directory up (the src directory)
        return os.path.dirname(utilities_path)

    @staticmethod
    def get_root_path(file=None):
        # The root directory is the parent of the source directory
        directory = os.path.dirname(Filesystem.get_source_path())

        if file is None:
            return directory

        return os.path.join(directory, file)

    @staticmethod
    def create(path):
        path_split = path.split('/')
        current_path = Filesystem.get_root_path()
        for path_fragment in path_split:
            current_path = os.path.join(current_path, path_fragment)
            if not os.path.isdir(current_path):
                os.makedirs(current_path)
