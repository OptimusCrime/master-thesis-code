#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import math

import numpy as np

from rorschach.prediction.tensorflow.tools import batch_gen
from rorschach.utilities import Config

with open(Config.get_path('path.data', 'training.json')) as data_file:
    training_set = np.array(json.load(data_file))

output = []
batch = batch_gen(training_set, training_set, Config.get('predicting.batch-size'))
iterations = num_batches = int(
            math.ceil(Config.get('preprocessing.training-set.size') / Config.get('predicting.batch-size')))

for i in range(iterations):
    batchX, batchY = batch.__next__()
    assert(len(batchX) == Config.get('predicting.batch-size'))
    output.extend(batchX)

assert(len(output) == Config.get('preprocessing.training-set.size'))

print('Duplicates: ', len(output) - len(set(output)))
