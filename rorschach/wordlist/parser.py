#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from random import randrange

from rorschach.utilities import Filesystem, LoggerWrapper


class WordListParser:

    def __init__(self):
        self.log = LoggerWrapper.load(__name__)
        self._words = None
        self.list_length = None

    def random_word(self, length=None):
        if self.list_length is None:
            self.run(length)

        return self.words[randrange(0, self.list_length)]

    @property
    def words(self):
        if self.list_length is None:
            self.run()

        return self._words

    def run(self, length=None):
        # Create first _words as a set to avoid duplicate entries
        self._words = set()

        files = self.get_word_files()
        self.get_words(files, length)

        # Change the set to a list to allow index lookup used by the randomizer
        self._words = list(self._words)

        self.list_length = len(self._words)

        self.log.info('Word lists have a total of %d words', self.list_length)

    def get_word_files(self):
        wordfiles_dir = Filesystem.get_root_path('config/wordlists')
        files = os.listdir(wordfiles_dir)
        files.remove('.gitignore')

        # We preappend all the files in the list with the path for the wordlists directory
        files = ['{0}/{1}'.format(wordfiles_dir, i) for i in files]

        # Log
        self.log.info('Found %d word lists', len(files))

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
