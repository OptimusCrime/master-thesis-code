#!/usr/bin/env python
# -*- coding: utf-8 -*-

class WidthCalculator:

    def __init__(self):
        pass

    @staticmethod
    def calculate(data_set):
        return data_set[0].shape[-1]
