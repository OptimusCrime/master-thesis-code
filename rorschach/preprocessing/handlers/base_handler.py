#!/usr/bin/env python
# -*- coding: utf-8 -*-


class BaseHandler:

    def __init__(self):
        self.set_list = []

    def add(self, pad_set):
        self.set_list.append(pad_set)

    def run(self):
        pass
