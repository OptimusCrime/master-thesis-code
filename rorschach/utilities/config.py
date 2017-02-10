#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import yaml

from rorschach.utilities import Filesystem


class Config:

    CONTENTS = None

    def __init__(self):
        pass

    @staticmethod
    def load_config():
        Config.CONTENTS = Config.load_config_file('config-default.yaml')

        override = Config.load_config_file('config.yaml')

        if override:
            Config.override_config(override)

        # Make sure that we found anything, or just quit life
        if Config.CONTENTS is None or Config.CONTENTS == '':
            sys.exit(1)

    @staticmethod
    def override_config(obj, path=''):
        for key, value in obj.items():
            if type(value) is dict:
                Config.override_config(value, path + '.' + key)
                continue

            Config.override_config_value(value, path + '.' + key)

    @staticmethod
    def override_config_value(value, path):
        # Remove the first part of the path as it is prefixed with '.'
        path_split = path.split('.')[1:]

        # Loop the current pool (instead of reccursion), begin with the outmost content dict
        current_pool = Config.CONTENTS
        for sub_key in path_split:
            # If we are currently handling the last part of the path we should update the value instead of
            # overwriting our pool
            if sub_key == path_split[-1]:
                if sub_key not in current_pool:
                    current_pool[sub_key] = None
                current_pool[sub_key] = value
                break

            current_pool = current_pool[sub_key]

    @staticmethod
    def load_config_file(file):
        if not os.path.exists(Filesystem.get_root_path('config/' + file)):
            return {}

        return yaml.safe_load(open(Filesystem.get_root_path('config/' + file)))

    @staticmethod
    def get(key):
        if Config.CONTENTS is None:
            Config.load_config()

        # Special handler for list of acceptable characters
        if key == 'general.characters':
            # Split all characters, strip whitespace, remove empty elements from the list
            return list(filter(None, [x.strip() for x in Config.CONTENTS['general']['characters'].split(',')]))

        # Shortcut for canvas size
        if key == 'preprocessing.canvas.size':
            return Config.CONTENTS['preprocessing']['canvas']['width'], \
                   Config.CONTENTS['preprocessing']['canvas']['height']

        # Special handler for the font file location
        if key == 'preprocessing.text.font':
            return Filesystem.get_root_path('fonts/' + Config.CONTENTS['preprocessing']['text']['font'] + '.ttf')

        # If we have no dash in our key we can access it directly
        if '.' not in key:
            return Config.CONTENTS[key]

        # Multilevel key. Traverse the tree
        return Config.get_nested(key)

    @staticmethod
    def get_nested(key):
        key_split = key.split('.')
        current_pool = Config.CONTENTS
        for sub_key in key_split:
            if sub_key not in current_pool:
                return None
            current_pool = current_pool[sub_key]
        return current_pool
