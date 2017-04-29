# -*- coding: utf-8 -*-

import json

from rorschach.utilities import Config


class DataContainer:

    def __init__(self):
        self.data = {
            'epoch': 0,
            'train_loss': [],
            'validate_loss': [],
            'validate_accuracy': [],
            'stores': []
        }

        self.offset = 0

        if Config.get('general.mode') == 'continue':
            with open(Config.get_path('path.output', 'data.json', fragment=Config.get('uid'))) as json_data:
                self.data = json.load(json_data)
                self.offset = self.data['epoch']

    def add(self, key, value):
        self.data[key] = value

    def set(self, key, value):
        if key == 'epoch' and Config.get('general.mode') == 'continue':
            self.add(key, value + self.offset)
        else:
            self.add(key, value)

    def remove(self, key):
        if key not in self.data:
            return

        del self.data[key]

    def create_list(self, key):
        self.data[key] = []

    def add_list(self, key, value):
        if key not in self.data or type(self.data[key]) != list:
            self.create_list(key)

        self.data[key].append(value)

    def all(self):
        return self.data

    def has(self, key):
        return key in self.data

    def get(self, key):
        # This may throw an exception. If it does, you messed up!
        return self.data[key]

    def reset(self, content):
        self.data = content
