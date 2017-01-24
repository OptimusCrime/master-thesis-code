from keras.layers import (GRU, Activation, Dense, Dropout, Masking, Merge,
                          TimeDistributed, Embedding, LSTM, Permute, Flatten, MaxPooling1D)
from keras.layers.convolutional import AveragePooling1D
from keras.models import Sequential, Model, Graph
from keras.utils.visualize_util import plot


track_input_shape = (8, 10)

g = Graph()
g.add_input(name='in1', input_shape=track_input_shape)
g.add_node(Masking(mask_value=0.,), name='mask1', input='in1')
#g.add_node(LSTM(128, return_sequences=True), name='forward1', input='mask1')

g.add_input(name='in2', input_shape=track_input_shape)
g.add_node(Masking(mask_value=0.,), name='mask2', input='in2')
#g.add_node(LSTM(128, return_sequences=True), name='forward2', input='mask2')

g.add_input(name='in3', input_shape=track_input_shape)
g.add_node(Masking(mask_value=0.,), name='mask3', input='in3')
#g.add_node(LSTM(128, return_sequences=True), name='forward3', input='mask3')

g.add_node(LSTM(128, return_sequences=True), name='int1', input='mask1')
g.add_node(LSTM(128, return_sequences=True), name='int2', inputs=['int1', 'mask2'])
g.add_node(LSTM(128, return_sequences=True), name='int3', inputs=['int2', 'mask3'])

#g.add_node(LSTM(128, return_sequences=True, go_backwards=True), name='backward', input='mask')
g.add_output(name='output1', input='int3', merge_mode='sum')

g.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

g.summary()

plot(g, to_file='test1232.png', show_shapes=True)
