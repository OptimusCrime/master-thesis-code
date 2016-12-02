#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utilities import Filesystem, unpickle_data

file_name = input('File name: ')

data = unpickle_data(Filesystem.get_root_path('data/' + file_name + '.pickl'))

print(' ')
print('Content of file is:')
print(data)
