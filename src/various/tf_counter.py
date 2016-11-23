#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import tensorflow as tf

seq_width = 20


def sequencify(x):
    seq = np.zeros(seq_width, dtype=np.int32)
    current_index = 0
    current_type = None
    current_length = 0
    for v in x:
        if current_type is None:
            current_type = v
            current_length = 1
            continue

        if current_type != v:
            seq[current_index] = current_length
            current_index += 1
            current_length = 1
            current_type = not current_type
            continue

        current_length += 1

    seq[current_index] = current_length

    # x will be a numpy array with the contents of the placeholder below
    # return np.array([0.1, 0.1, 0.1])
    return np.array(seq, dtype=np.int32)


class Model:

    def __init__(self):
        pass


x = tf.placeholder(tf.bool, None)
y = tf.py_func(sequencify, [x], [tf.int32])

with tf.Session() as session:
    tf.initialize_all_variables().run()

    result = session.run(y, feed_dict={x: [True, False, False, False, True, False, False, False, False, True]})
    print(result)
