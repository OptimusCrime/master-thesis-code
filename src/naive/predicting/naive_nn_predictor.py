#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .abstracts import AbstractPredictor
from utilities import Config

import numpy as np
import tensorflow as tf


class NaiveNNPredictor(AbstractPredictor):

    def __init__(self):
        super().__init__()

        self.widest = None

        # Storing the transformed data
        self.training_images_transformed = None
        self.training_labels_transformed = None
        self.phrase_transformed = None

    @staticmethod
    def from_character_to_character_index(char):
        characters = Config.get('characters')
        for ii in range(len(characters)):
            if characters[ii] == char:
                return ii
        return -1

    def preprocess(self):
        self.calculate_widest()

        self.transform_data_set()
        self.transform_phrase()

    def calculate_widest(self):
        # Get the widest data set
        for data in self.data_set:
            data_flatten = data['matrix'].flatten()
            if self.widest is None or len(data_flatten) > self.widest:
                self.widest = len(data_flatten)

    def transform_data_set(self):
        # Transform and create the data set with corresponding labels
        self.training_images_transformed = np.ones((len(self.data_set), self.widest))
        self.training_labels_transformed = np.zeros((len(self.data_set), len(self.data_set)))

        for i in range(len(self.data_set)):
            data_flatten = self.data_set[i]['matrix'].flatten()
            for j in range(len(data_flatten)):
                if data_flatten[j] == 0:
                    self.training_images_transformed[i][j] = 0
            character_index = NaiveNNPredictor.from_character_to_character_index(self.data_set[i]['character'])
            if character_index >= 0:
                self.training_labels_transformed[i][character_index] = 1

    def transform_phrase(self):
        self.phrase_transformed = self.phrase[0]['matrix'][0]

    def predict(self):
        learning_rate = 0.01
        training_epochs = 400

        size_input = self.widest
        size_output = len(Config.get('characters'))

        # tf Graph Input
        x = tf.placeholder(tf.float32, [None, size_input], name='Input')
        y = tf.placeholder(tf.float32, [None, size_output], name='Output')

        # Set model weights
        weights = tf.Variable(tf.zeros([size_input, size_output]))
        b = tf.Variable(tf.zeros([size_output]))

        # Construct model
        pred = tf.nn.softmax(tf.matmul(x, weights) + b)  # Softmax

        # Minimize error using cross entropy
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(training_epochs):
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: self.training_images_transformed,
                                                              y: self.training_labels_transformed})

                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            for i in range(0, len(self.phrase_transformed)):
                padded = self.phrase_transformed[i: min(i + size_input, len(self.phrase_transformed))]
                ipt = np.pad(padded, (0, self.widest - len(padded)), 'constant', constant_values=1)

                # Only predict when we have a starting 0
                if ipt[0] == 1:
                    continue

                # If the previous pixel also was a 0 we can ignore this as it is part of another signature
                if i != 0 and self.phrase_transformed[i - 1] == 0:
                    continue

                # If the last pixel is a 0 and we have a 0 following it, we can ignore it as a part of another signature
                if ipt[len(ipt) - 1] == 0 and (i + size_input) < len(self.phrase_transformed) and \
                        self.phrase_transformed[i + size_input] == 0:
                    continue

                # Do the actual prediction here
                prediction = sess.run(pred, {
                    x: np.array([ipt], dtype=np.float)
                })

                # Add to list of predictions
                self.predictions.append({
                    'input': ipt,
                    'offset': i,
                    'values': prediction
                })
