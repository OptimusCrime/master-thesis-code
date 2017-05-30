#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

from rorschach.utilities import Config, pickle_data, unpickle_data


class DatasetSelector:

    def __init__(self):
        self.words = [797]
        self.individual_length = 10000

    def run(self):
        data = unpickle_data(Config.get_path('path.data', 'test_set.pickl'))
        new_dataset = []

        for word in self.words:
            for _ in range(self.individual_length):
                new_dataset.append(copy.deepcopy(data[word]))

        pickle_data(new_dataset, Config.get_path('path.data', 'test_set_new.pickl'))


if __name__ == '__main__':
    data_selector = DatasetSelector()
    data_selector.run()
