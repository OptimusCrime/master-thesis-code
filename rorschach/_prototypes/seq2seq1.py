from keras.layers import (GRU, Activation, Dense, Dropout, Masking, Merge,
                          TimeDistributed, Embedding, LSTM, Permute)
from keras.layers.convolutional import AveragePooling1D
from keras.models import Sequential, Model
from keras.utils.visualize_util import plot

#left = Sequential()
#left.add(Masking(mask_value=0.,
#             input_shape=(self.widest, 1)))

model = Sequential()
model.add(LSTM(input_shape=(24, 8),
                output_dim=256,
                activation='sigmoid',
                inner_activation='hard_sigmoid',
                return_sequences=True))

model.add((LSTM(256, return_sequences=True)))
model.add((LSTM(128, return_sequences=True)))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(18)))

model.add(Permute((2, 1), input_shape=(24, 18)))
model.add(TimeDistributed(Dense(16)))
model.add(Permute((2, 1), input_shape=(16, 18)))

#model.add(AveragePooling1D(pool_length=self.pooling_factor))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()

plot(model, to_file='test123.png', show_shapes=True)
