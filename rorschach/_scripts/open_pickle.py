#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.utilities import Config, unpickle_data

data = unpickle_data(Config.get_path('path.data', 'test_set.pickl'))

print(data)
