from keras.layers import (GRU, Activation, Dense, Dropout, Masking, Merge,
                          TimeDistributed, Embedding, LSTM, Permute, Flatten, MaxPooling1D, Input)
from keras.layers.convolutional import AveragePooling1D
from keras.models import Sequential, Model
from keras.utils.visualize_util import plot

from rorschach.prediction.layer import HiddenStateLSTM

inputs = []
encoders = []
hidden = []

for i in range(5):
    name_input = 'in' + str(i)
    embedding_name = 'embedding' + str(i)
    encoder_intermediate = 'encoder' + str(i)

    current_input = Input(shape=(10,), name=name_input)
    current_embedding = Embedding(300, 19, mask_zero=False, name=embedding_name)(current_input)

    current_encoder = None
    current_hidden = None
    if i == 0:
        # First input, has only one input
        current_encoder, *current_hidden = HiddenStateLSTM(128, dropout_W=0.5, dropout_U=0.5, return_sequences=False)(current_embedding)
    else:
        if i == 4:
            # Input 5, has two inputs like the rest, but also returns the entire sequence
            current_encoder, _, _ = HiddenStateLSTM(128, dropout_W=0.5, dropout_U=0.5,
                                                               return_sequences=True)([current_embedding] + hidden[-1])

        else:
            # Input 2 - 4 has two inputs, the input and the previous LSTM
            current_encoder, *current_hidden = HiddenStateLSTM(128, dropout_W=0.5, dropout_U=0.5, return_sequences=False)(
                [current_embedding] + hidden[-1])

    inputs.append(current_input)
    encoders.append(current_encoder)

    if current_hidden is not None:
        hidden.append(current_hidden)

decoders = []
decoder_hidden = []
outputs = []
for i in range(10):
    if i == 0:
        # Input the output of the encoders
        current_decoder, *current_decoder_hidden = HiddenStateLSTM(128, dropout_W=0.5, dropout_U=0.5, return_sequences=True)(encoders[-1])
    else:
        current_decoder, *current_decoder_hidden = HiddenStateLSTM(128, dropout_W=0.5, dropout_U=0.5,
                                                                   return_sequences=True)([decoders[-1]] + decoder_hidden[-1])

    current_inner_decoder = LSTM(128, dropout_W=0.5, dropout_U=0.5, return_sequences=False)(current_decoder)
    current_output = Dense(19)(current_inner_decoder)
    current_output = Dropout(0.2)(current_output)
    current_output = Activation('softmax')(current_output)

    decoders.append(current_decoder)
    decoder_hidden.append(current_decoder_hidden)
    outputs.append(current_output)

model = Model(input=inputs, output=outputs)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

plot(model, to_file='test1232.png', show_shapes=True)
