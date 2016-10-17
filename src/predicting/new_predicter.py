#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utilities import Config, Filesystem, unpickle_data

import numpy as np
import tensorflow as tf

# WIP
# WIP
# WIP

def from_character_to_character_index(char):
    characters = Config.get('characters')
    for i in range(len(characters)):
        if characters[i] == char:
            return i
    return -1


# Load data set
raw_data_set = unpickle_data(Filesystem.get_root_path('data/data_set.pickl'))

# Get the widest data set
widest = None
for data in raw_data_set:
    data_flatten = data['matrix'].flatten()
    if widest is None or len(data_flatten) > widest:
        widest = len(data_flatten)

# Transform and create the data set with corresponding labels
data_set = np.ones((len(raw_data_set), widest))
labels = np.zeros((len(raw_data_set), len(raw_data_set)))

for i in range(len(raw_data_set)):
    data_flatten = raw_data_set[i]['matrix'].flatten()
    for j in range(len(data_flatten)):
        if data_flatten[j] == 0:
            data_set[i][j] = 0
    character_index = from_character_to_character_index(raw_data_set[i]['character'])
    if character_index >= 0:
        labels[i][character_index] = 1

print(data_set)

# Here goes Tensorflow!
learning_rate = 0.01
training_epochs = 70

size_input = widest
size_output = len(Config.get('characters'))

# tf Graph Input
x = tf.placeholder(tf.float32, [None, size_input], name='Input') # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, size_output], name='Output') # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([size_input, size_output]))
b = tf.Variable(tf.zeros([size_output]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

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
        _, c = sess.run([optimizer, cost], feed_dict={x: data_set,
                                                      y: labels})

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: data_set, y: labels}))

    print('')
    print('Predicting time!')
    print('')

    raw_phrase = unpickle_data(Filesystem.get_root_path('data/phrase.pickl'))
    print(raw_phrase)
    phrase = raw_phrase[0]['matrix'][0]
    print(phrase)

    for i in range(0, len(phrase)):
        padded = phrase[i : min(i + size_input, len(phrase))]
        ipt = np.pad(padded, (0, 17 - len(padded)), 'constant',  constant_values=1)
        print(ipt)

        # Only predict when we have a starting 0
        if ipt[0] == 1:
            continue

        # If the previous pixel also was a 0 we can ignore this as it is part of another signature
        if i != 0 and phrase[i - 1] == 0:
            continue

        # If the last pixel is a 0 and we have a 0 following it, we can ignore it as a part of another signature
        if ipt[len(ipt) - 1] == 0 and i < len(phrase) and phrase[i + size_input] == 0:
            continue

        print('Predicting: ')
        print(ipt)

        feed_dict = {
            x: np.array([ipt], dtype=np.float)
        }

        prediction = sess.run(pred, feed_dict)
        print(prediction)
        print(np.argmax(prediction))
        print(prediction[0][np.argmax(prediction)])
        print(Config.get('characters')[np.argmax(prediction)])
        print('-----')
