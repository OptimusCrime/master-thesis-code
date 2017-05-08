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

        # Go one directory up (the rorchach directory)
        return os.path.dirname(utilities_path)

    @staticmethod
    def get_root_path(file=None):
        # The root directory is the parent of the source directory
        directory = os.path.dirname(Filesystem.get_source_path())

        if file is None:
            return directory

        return os.path.join(directory, file)

    @staticmethod
    def save(path, name):
        Filesystem.create(path, outside=True)
        return os.path.join(path, name)

    @staticmethod
    def create(path, outside=False):
        path_split = path.split(os.sep)
        current_path = Filesystem.get_root_path()
        path_rebuilt = []

        for path_fragment in path_split:
            if not outside:
                current_path = os.path.join(current_path, path_fragment)
            else:
                path_rebuilt.append(path_fragment)
                current_path = str(os.sep).join(path_rebuilt)

                # Fix for Unix type systems where the path begins with a leading /
                if current_path == '':
                    continue

            if not os.path.isdir(current_path):
                os.makedirs(current_path)
