# -*- coding: utf-8 -*-

from abc import ABCMeta

from rorschach.common import DataSetTypes
from rorschach.preprocessing.handlers import ConstraintHandler
from rorschach.utilities import Config, LoggerWrapper


class AbstractCreator:
    __metaclass__ = ABCMeta

    def __init__(self, type):
        self.log = LoggerWrapper.load(__name__)

        self.type = type
        self.contents = []
        self.constraint_handler = ConstraintHandler()
        self.terms = []
        self.fonts = Config.get('preprocessing.text.fonts')

    @property
    def set_type_keyword(self):
        return DataSetTypes.type_to_keyword(self.type)

    def create(self):
        self.contents = self.create_sets()
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
