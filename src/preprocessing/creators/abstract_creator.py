#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta

import numpy as np

from preprocessing.handlers import ConstraintHandler
from utilities import Config, LoggerWrapper


class AbstractCreator:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.log = LoggerWrapper.load(__name__)

        self.contents = []
        self.constraint_handler = ConstraintHandler()

    def create(self):
        self.create_sets()
        self.apply_constraints()
        self.apply_signature()

    def create_sets(self):
        pass

    def apply_constraints(self):
        pass

    def apply_signature(self):
        signature_position = Config.get('preprocessing.signature.position')
        signature_height = Config.get('preprocessing.signature.height')

        for i in range(len(self.contents)):
            self.contents[i]['matrix'] = self.contents[i]['matrix'][
                                         signature_position:(signature_position + signature_height)]

    def save(self):
        pass
