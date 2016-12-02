#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from rorschach.utilities import Config
from rorschach.wordbuilder.parser import ListParser


class ProbabilityParser:
    def __init__(self):
        self.char_map = {}
        self.words = []
        self.prob_initial = None
        self.prob_trans = None

    def run(self):
        list_parser = ListParser()
        self.words = list_parser.words

        self.build_char_hash_map()

        self.calculate_probability_initial()
        self.calculate_probability_state()

    def build_char_hash_map(self):
        chars = Config.get('characters')
        for i in range(len(chars)):
            self.char_map[chars[i]] = i

    def calculate_probability_initial(self):
        self.prob_initial = np.zeros((len(self.char_map)))

        count_initial = np.zeros((len(self.char_map)), dtype=np.int)

        for word in self.words:
            letter = word[0]
            count_initial[self.char_map[letter]] += 1

        number_of_words = len(self.words)

        for i in range(len(self.prob_initial)):
            self.prob_initial[i] = count_initial[i] / number_of_words

    def calculate_probability_state(self):
        self.prob_trans = np.zeros((len(self.char_map), len(self.char_map)))

        count_trans = np.zeros((len(self.char_map), len(self.char_map)), dtype=np.int)
        trans = np.zeros((len(self.char_map)), dtype=np.int)

        for word in self.words:
            for i in range(1, len(word)):
                # count_trans[from][to] += 1
                idx_from = self.char_map[word[i - 1]]
                idx_to = self.char_map[word[i]]
                count_trans[idx_from][idx_to] += 1

                # Keep track of how many translations there are in total
                trans[idx_from] += 1

        for i in range(len(count_trans)):
            for j in range(len(count_trans)):
                self.prob_trans[i][j] = count_trans[i][j] / trans[i]
