#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random

from rorschach.utilities import Filesystem


class WordListParser:

    def __init__(self):
        self._words = set()

    def random_word(self, length=None):
        if len(self._words) == 0:
            self.run(length)

        return random.choice(tuple(self.words))

    @property
    def words(self):
        if len(self._words) == 0:
            self.run()

        return self._words

    def run(self, length=None):
        files = WordListParser.get_word_files()
        self.get_words(files, length)

    @staticmethod
    def get_word_files():
        wordfiles_dir = Filesystem.get_root_path('config/wordlists')
        files = os.listdir(wordfiles_dir)
        files.remove('.gitignore')

        # We preappend all the files in the list with the path for the wordlists directory
        files = ['{0}/{1}'.format(wordfiles_dir, i) for i in files]

        return files

    def get_words(self, files, length=None):
        for file in files:
            self.get_words_from_file(file, length)

    def get_words_from_file(self, file, length=None):
        lines = [line.rstrip('\n') for line in open(file)]
        for line in lines:
            self.validate_string(line, length)

    def validate_string(self, line, length=None):
        if line.isalpha() and len(line) > 1:
            if length is None or len(line) <= length:
                self._words.add(line.upper())