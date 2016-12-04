#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

from rorschach.utilities import Config, ModuleImporter


class Transformator:

    def __init__(self):
        self.input_lists = []
        self.output_lists = {}

    def run(self):
        self.construct_lists()
        self.get_handlers()

    def construct_lists(self):
        for data_list in self.input_lists:
            self.output_lists[data_list['type']] = {
                'images': [],
                'labels': []
            }

    def get_handlers(self):
        handlers = Config.get('transformation.handlers')

        current_input = self.input_lists

        for handler_module_path in handlers:
            handler_module = ModuleImporter.load(handler_module_path)

            if handler_module is not None:
                current_input = handler_module.run(copy.deepcopy(current_input))

        # TODO add current input lists to output lists
        print(current_input)

    def get_data_set(self, set_type):
        if set_type not in self.output_lists:
            return [], []

        data_list = self.output_lists[set_type]
        return data_list['images'], data_list['labels']
