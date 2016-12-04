#!/usr/bin/env python
# -*- coding: utf-8 -*-


class BaseHandler:

    def __init__(self):
        pass

    def run(self, input_lists):
        for key in input_lists:
            self.list_handler(input_lists[key], key)

    def list_handler(self, input_list, key):
        pass

    def obj_handler(self, obj):
        pass
