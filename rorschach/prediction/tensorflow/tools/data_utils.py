#!/usr/bin/env python
# -*- coding: utf-8 -*-


def batch_gen(x, y, batch_size):
    if (len(x) % batch_size) != 0:
        raise Exception('Batch must be dividable on batch_size')

    while True:
        for i in range(0, len(x), batch_size):
            yield x[i:(i + batch_size)].T, y[i:(i + batch_size)].T
