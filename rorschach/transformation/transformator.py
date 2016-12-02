#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.utilities import Config, ModuleImporter


class Transformator:

    def __init__(self):
        self.data_lists = []
        self.data_lists_transformed = {}

    def run(self):
        self.construct_lists()
        self.get_handlers()

    def construct_lists(self):
        for data_list in self.data_lists:
            self.data_lists_transformed[data_list['type']] = {
                'images': [],
                'labels': []
            }

    def get_handlers(self):
        handlers = Config.get('transformation.handlers')
        for handler_module_path in handlers:
            handler_module = ModuleImporter.load(handler_module_path)

            if handler_module is not None:
                handler_module.run(self.data_lists, self.data_lists_transformed)

    def get_data_set(self, type):
        if type not in self.data_lists_transformed:
            return [], []

        data_list = self.data_lists_transformed[type]
        return data_list['images'], data_list['labels']
