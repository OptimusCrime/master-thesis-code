#!/usr/bin/env python
# -*- coding: utf-8 -*-


class BaseHandler:

    DATA_SET = 0
    PHRASE = 1
    WORD_SET = 2

    def __init__(self):
        self.set_list = []

    def add(self, pad_set):
        self.set_list.append(pad_set)

    def run(self):
        pass
