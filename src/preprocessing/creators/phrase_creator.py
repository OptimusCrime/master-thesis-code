#!/usr/bin/env python
# -*- coding: utf-8 -*-

from preprocessing.creators import TermCreator
from utilities import Filesystem, pickle_data


class PhraseCreator(TermCreator):

    def save(self):
        pickle_data(self.contents, Filesystem.get_root_path('data/phrase.pickl'))
