#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utilities import Filesystem, unpickle_data



data_set = unpickle_data(Filesystem.get_root_path('data/data_set.pickl'))

widest = None

for data in data_set:
    data_flatten = data['matrix'].flatten()
    if widest is None or len(data_flatten) > widest:
        widest = len(data_flatten)
        print(data_flatten)
print(widest)
