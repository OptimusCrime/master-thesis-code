#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from random import randrange

from rorschach.utilities import Config, Filesystem, LoggerWrapper


class WordListParser:

    def __init__(self):
        self.log = LoggerWrapper.load(__name__)
        self._words = None
        self.list_length = None

        # Keep track of duplicates
        self.duplicates_all = []
        self.duplicates_set = {}

    def random_word(self, word_list=None):
        if self.list_length is None:
            self.run(Config.get('preprocessing.input.max-length'))

        remove_duplicate_set = Config.get('wordlist.remove-duplicate-set')
        remove_duplicate_all = Config.get('wordlist.remove-duplicate-all')

        # If both settings are false we can just return a random word
        if not remove_duplicate_set and not remove_duplicate_all:
            return self.words[randrange(0, self.list_length)]

        # Duplicate all has precedence over duplicate set
        while True:
            random_word = self.words[randrange(0, self.list_length)]

            if self.not_duplicated_word(random_word, remove_duplicate_set, remove_duplicate_all, word_list=word_list):
                return random_word

    def not_duplicated_word(self, word, remove_duplicate_set, remove_duplicate_all, word_list=None):
        if remove_duplicate_all:
            return self.not_duplicated_word_all(word)

        if word_list is None:
            raise Exception('Filtering duplicate words on word set, but no word set was provided')

        return self.not_duplicated_word_set(word, word_list)

    def not_duplicated_word_all(self, word):
        if word not in self.duplicates_all:
            self.duplicates_all.append(word)
            return True
        return False

    def not_duplicated_word_set(self, word, word_list):
        # Make sure to create a list in the dict for our set
        if word_list not in self.duplicates_set:
            self.duplicates_set[word_list] = []

        if word not in self.duplicates_set[word_list]:
            self.duplicates_set[word_list].append(word)
            return True
        return False

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
