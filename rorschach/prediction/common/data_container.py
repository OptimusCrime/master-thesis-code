# -*- coding: utf-8 -*-


class DataContainer:

    def __init__(self):
        self.data = {
            'epoch': 0,
            'train_loss': [],
            'validate_loss': [],
            'validate_accuracy': [],
            'stores': []
        }

    def add(self, key, value):
        self.data[key] = value

    def set(self, key, value):
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
