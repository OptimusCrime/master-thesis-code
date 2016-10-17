#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

from utilities import Config, Filesystem


class ListParser:

    def __init__(self):
        self._words = set()

    @property
    def words(self):
        if len(self._words) == 0:
            self.run()

        return self._words

    def run(self):
        files = self.get_word_files()
        self.get_words(files)

    def get_word_files(self):
        wordfiles_dir = Filesystem.get_root_path('config/wordlists')
        files = os.listdir(wordfiles_dir)
        files.remove('.gitignore')

        # We preappend all the files in the list with the path for the wordlists directory
        files = ['{0}/{1}'.format(wordfiles_dir, i) for i in files]

        return files

    def get_words(self, files):
        for file in files:
            self.get_words_from_file(file)

    def get_words_from_file(self, file):
        lines = [line.rstrip('\n') for line in open(file)]
        for line in lines:
            self.validate_string(line)

    def validate_string(self, line):
        if line.isalpha() and len(line) > 1:
            self._words.add(line.upper())
