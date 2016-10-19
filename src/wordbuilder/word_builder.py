#!/usr/bin/env python
# -*- coding: utf-8 -*-

from wordbuilder.parser import ProbabilityParser


class WordBuilder:
    def __init__(self):
        pass

    @staticmethod
    def run():
        probability_parser = ProbabilityParser()
        probability_parser.run()
