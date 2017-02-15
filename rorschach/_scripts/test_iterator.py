#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from rorschach.prediction.tensorflow.tools import batch_gen

trainX = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
trainY = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

testX = np.array([20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40])
testY = np.array([40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80])

train_gen = batch_gen(trainX, trainY, 2)
test_gen = batch_gen(testX, testY, 2)

for i in range(10):
    val_trainX, val_trainY = train_gen.__next__()
    val_testX, val_testY = test_gen.__next__()

    print(val_trainX)
    print(val_testX)
    print('---')

