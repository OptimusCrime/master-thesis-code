#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.utilities import Config, unpickle_data

data = unpickle_data(Config.get_path('path.output', 'context_data.pickl', fragment=Config.get('uid')))
print([x for x, value in data.items()])

print(data['MYTHIC'][0])

print([len(value) for x, value in data.items()])
