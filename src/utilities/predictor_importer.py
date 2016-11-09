#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import sys


class PredictorImporter:

    def __init__(self):
        pass

    @staticmethod
    def load(predictor_module_string):
        try:
            module_path_split = predictor_module_string.split('.')

            # We're transforming path.to.module_name to the format "from path.to import module_name", which is the
            # format we're using
            module_path = '.'.join(module_path_split[:-1])
            model_name = module_path[-1]

            return importlib.import_module(module_path, model_name)
        except:
            print('Could not import predictor ' + predictor_module_string, file=sys.stderr)

        return None
