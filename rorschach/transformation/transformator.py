#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

from rorschach.utilities import Config, ModuleImporter


class Transformator:

    def __init__(self):
        self.original_lists = []
        self.transformed_lists = {}

    def run(self):
        self.get_handlers()

    def construct_lists(self, original_lists):
        self.original_lists = original_lists

        for data_list in self.original_lists:
            self.transformed_lists[data_list['type']] = {
                'images': data_list['set'],
                'labels': []
            }

    def get_handlers(self):
        handlers = Config.get('transformation.handlers')

        current_input = self.transformed_lists

        for handler_module_path in handlers:
            handler_module = ModuleImporter.load(handler_module_path)

            if handler_module is not None:
                current_input = handler_module.run(copy.deepcopy(current_input))

        # TODO add current input lists to output lists
        print(current_input)
        print('------')

        self.transformed_lists = current_input

    def get_data_set(self, set_type):
        if set_type not in self.transformed_lists:
            return [], []

        data_list = self.transformed_lists[set_type]
        return data_list['images'], data_list['labels']
