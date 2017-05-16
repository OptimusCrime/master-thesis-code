#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from random import randint, randrange

from rorschach.utilities import Config, Filesystem, LoggerWrapper


class WordListParser:

    LIST_LENGTH = None
    WORDS = []

    # Keep track of duplicates
    DUPLICATES_ALL = []
    DUPLICATES_SET = {}

    def __init__(self):
        self.log = LoggerWrapper.load(__name__)

    def random_word(self, word_list=None):
        if WordListParser.LIST_LENGTH is None:
            self.run(Config.get('preprocessing.input.max-length'))

        remove_duplicate_set = Config.get('wordlist.remove-duplicate-set')
        remove_duplicate_all = Config.get('wordlist.remove-duplicate-all')

        # If both settings are false we can just return a random word
        if not remove_duplicate_set and not remove_duplicate_all:
            return self.pick_random_word()

        # Duplicate all has precedence over duplicate set
        while True:
            random_word = self.pick_random_word()

            if self.not_duplicated_word(random_word, remove_duplicate_set, remove_duplicate_all, word_list=word_list):
                return random_word

    def pick_random_word(self):
        word = self.words[randrange(0, WordListParser.LIST_LENGTH)]
        if Config.get('preprocessing.random-upper-lower') in [None, False]:
            return word

        if randint(0, 1) == 0:
            return word

        return word.lower()

    def not_duplicated_word(self, word, remove_duplicate_set, remove_duplicate_all, word_list=None):
        if remove_duplicate_all:
            return self.not_duplicated_word_all(word)

        if word_list is None:
            raise Exception('Filtering duplicate words on word set, but no word set was provided')

        return self.not_duplicated_word_set(word, word_list)

    def not_duplicated_word_all(self, word):
        if word not in WordListParser.DUPLICATES_ALL:
            WordListParser.DUPLICATES_ALL.append(word)
            return True
        return False

    def not_duplicated_word_set(self, word, word_list):
        # Make sure to create a list in the dict for our set
        if word_list not in WordListParser.DUPLICATES_SET:
            WordListParser.DUPLICATES_SET[word_list] = []

        if word not in WordListParser.DUPLICATES_SET[word_list]:
            WordListParser.DUPLICATES_SET[word_list].append(word)
            return True
        return False

    @property
    def words(self):
        if WordListParser.LIST_LENGTH is None:
            self.run()

        return WordListParser.WORDS

    def run(self, length=None):
        # Create first _words as a set to avoid duplicate entries
        WordListParser.WORDS = set()

        files = self.get_word_files()
        self.get_words(files, length)

        # Change the set to a list to allow index lookup used by the randomizer
        WordListParser.WORDS = list(WordListParser.WORDS)

        WordListParser.LIST_LENGTH = len(WordListParser.WORDS)

        self.log.info('Word lists have a total of %d words', WordListParser.LIST_LENGTH)

    def get_word_files(self):
        wordfiles_dir = Filesystem.get_root_path('config/wordlists')
        files = os.listdir(wordfiles_dir)
        if os.path.exists(wordfiles_dir + os.sep + '.gitignore'):
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
                WordListParser.WORDS.add(line.upper())
