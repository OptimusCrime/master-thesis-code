#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.utilities.filesystem import Filesystem

import os
import sys
import yaml


class Config:

    CONTENTS = None

    def __init__(self):
        pass

    @staticmethod
    def load_config():
        # Fallback to the config example file
        file = 'config.yaml'
        if not os.path.exists(Filesystem.get_root_path('config/config.yaml')):
            file = 'config_example.yaml'

        # Load the contents
        Config.CONTENTS = yaml.safe_load(open(Filesystem.get_root_path('config/' + file)))

        # Make sure that we found anything, or just quit life
        if Config.CONTENTS is None or Config.CONTENTS == '':
            sys.exit(1)

    @staticmethod
    def get(key):
        if Config.CONTENTS is None:
            Config.load_config()

        # Special handler for list of acceptable characters
        if key == 'characters':
            # Split all characters, strip whitespace, remove empty elements from the list
            return list(filter(None, [x.strip() for x in Config.CONTENTS['characters'].split(',')]))

        # Shortcut for canvas size
        if key == 'canvas_size':
            return (Config.CONTENTS['canvas']['width'], Config.CONTENTS['canvas']['height'])

        # Special handler for the font file location
        if key == 'text-font':
            return Filesystem.get_root_path('fonts/' + Config.CONTENTS['text']['font'] + '.ttf')

        # If we have no dash in our key we can access it directly
        if '-' not in key:
            return Config.CONTENTS[key]

        # Multilevel key. Traverse the tree
        key_split = key.split('-')
        current_pool = Config.CONTENTS
        for sub_key in key_split:
            current_pool = current_pool[sub_key]
        return current_pool
