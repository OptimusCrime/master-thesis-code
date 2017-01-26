#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.common import DataSetTypes


class BaseHandler:

    def __init__(self):
        self.input_lists = None

    def run(self, input_lists):
        self.input_lists = input_lists

        for key in input_lists:
            self.list_handler(input_lists[key], key)

    def list_handler(self, input_list, key):
        if not input_list[DataSetTypes.IMAGES]:
            return

        for i in range(len(input_list[DataSetTypes.IMAGES])):
            ipt = input_list[DataSetTypes.IMAGES][i]
            label = input_list[DataSetTypes.LABELS][i]
            self.obj_handler(ipt, label)

    def obj_handler(self, ipt, label):
        pass
