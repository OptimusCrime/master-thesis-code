#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorchach.preprocessing.creators import TermCreator
from rorchach.preprocessing.savers import MatrixSaver
from rorchach.utilities import Config, Filesystem, pickle_data


class PhraseCreator(TermCreator):

    def apply_constraints(self):
        super().apply_constraints()

        if Config.get('preprocessing.save.phrase'):
            MatrixSaver.save('data/phrase', self.contents)

    def save(self):
        if Config.get('preprocessing.save.phrase-signatures'):
            MatrixSaver.save('data/phrase-signatures', self.contents)

        pickle_data(self.contents, Filesystem.get_root_path('data/phrase.pickl'))
