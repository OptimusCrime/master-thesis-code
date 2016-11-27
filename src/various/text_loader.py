#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
import sys

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = 1
        self.seq_length = 10
        self.encoding = encoding

        input_file = os.path.join(data_dir, "inputt.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)) or 1 == 1:
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)
        print(counter)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        print(count_pairs)
        self.chars, _ = zip(*count_pairs)
        print('chars', self.chars)
        self.vocab_size = len(self.chars)
        print('vocab_size', self.vocab_size)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        print('vocab', self.vocab)
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)

        #print('vocab.get', list(self.vocab.get))
        print(data)
        print(list(map(self.vocab.get, data)))
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        #self.vocab_size = len(self.chars)
        self.vocab_size = 10
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small, let's give them a better error message
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        print(xdata)

        ydata = np.copy(self.tensor)
        print('ydata')
        print(ydata)
        ydata[:-1] = xdata[1:]
        print('xdata[1:]')
        print(xdata[1:])
        print('ydata[:-1]')
        print(ydata[:-1])

        ydata[-1] = xdata[0]
        print(ydata)
        #self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        #self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)


        #print(self.x_batches)
        #print(self.y_batches)

        self.x_batches = []
        self.y_batches = []

        self.x_batches.append(np.array([[3, 9, 3, 7, 14, 7, 3, 18, 3, 16, 3, 11, 3, 11, 3, 4, 0, 0, 0, 0]], dtype=np.int32))
        self.y_batches.append(np.array([[6, 5, 2, 2, 10, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32))

        print(len(self.y_batches))
        print(self.x_batches[0].shape)
        print(self.y_batches[0].shape)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer = 0
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
