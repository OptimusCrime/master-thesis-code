#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np


from rorschach.utilities import Config, Filesystem, unpickle_data

#file_name = input('File name: ')

data = unpickle_data(Config.get_path('path.output', 'predictions.pickl', fragment="confusionmatrix_test123"))

print(data)

for i in range(len(data['correct'])):
    for j in range(len(data['correct'][i])):
        print(np.argmax(data['correct'][i][j]))
        print(np.argmax(data['predictions'][i][j]))
        print('')
    print('###')

#print(' ')
#print('Content of file is:')
#print(data)
