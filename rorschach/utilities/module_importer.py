# -*- coding: utf-8 -*-

import importlib
import sys


class ModuleImporter:

    def __init__(self):
        pass

    @staticmethod
    def load(module_string):
        try:
            module_path_split = module_string.split('.')

            # We're transforming path.to.module_name to the format "from path.to import module_name", which is the
            # format we're using
            module_path = '.'.join(module_path_split[:-1])
            model_name = module_path_split[-1]

            # Actual handling of the import. Creates a new instace of the class. Code from:
            # http://stackoverflow.com/a/4821120/921563
            module = importlib.import_module(module_path, model_name)
            return getattr(module, model_name)()

        except Exception as e:
            print('Could not import module ' + module_string, file=sys.stderr)

            # Reraise previous exception
            raise
