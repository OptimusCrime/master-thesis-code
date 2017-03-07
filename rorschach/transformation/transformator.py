#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

from rorschach.common import DataSetTypes
from rorschach.utilities import Config, LoggerWrapper, ModuleImporter


class Transformator:

    def __init__(self, handler):
        self.log = LoggerWrapper.load(__name__)

        self.handlers = handlers
        self.data = {}
        self.original_lists = []
        self.transformed_lists = {}

    def run(self):
        self.get_handlers()

    def construct_lists(self, original_lists):
        self.original_lists = original_lists

        for data_list in self.original_lists:
            self.transformed_lists[data_list['type']] = data_list['set']

    def get_handlers(self):
        current_input = self.transformed_lists

        for handler_module_path in self.handlers:
            handler_module = ModuleImporter.load(handler_module_path)

            if handler_module is not None:
                self.log.info('Running transformator %s', handler_module_path)

                # Set data
                handler_module.data = self.data

                # Prepare
                handler_module.prepare()

                # Run the handler
                current_input = handler_module.run(copy.deepcopy(current_input))

        self.transformed_lists = current_input

    def data_set(self, set_type):
        if set_type not in self.transformed_lists:
            return [], []

        data_list = self.transformed_lists[set_type]
        return data_list[DataSetTypes.IMAGES], data_list[DataSetTypes.LABELS]
