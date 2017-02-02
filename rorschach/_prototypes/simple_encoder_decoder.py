from keras.layers import (GRU, Activation, Dense, Dropout, Masking, Merge,
                          TimeDistributed, LSTM, Permute, Bidirectional, Embedding, Input, RepeatVector)
from keras.layers.convolutional import AveragePooling1D
from keras.regularizers import WeightRegularizer, ActivityRegularizer
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.utils.visualize_util import plot

from rorschach.prediction.callbacks import CallbackWrapper
from rorschach.prediction.callbacks.plotter import PlotCallback
from rorschach.prediction.helpers import (EmbeddingCalculator,
                                          WidthCalculator)
from rorschach.prediction.layer import HiddenStateLSTM, HiddenStateLSTM2
from rorschach.prediction.nets import BasePredictor
from rorschach.utilities import Config, Filesystem, LoggerWrapper, unpickle_data  # isort:skip

enc_input = Input(shape=(48,), dtype='int32', name='encoder_input')
enc_layer = Embedding(300, 19, mask_zero=False)(enc_input)
enc_layer, *hidden = HiddenStateLSTM2(
    1024,
    dropout_W=0.2,
    dropout_U=0.2,
    return_sequences=False
)(enc_layer)

repeat = RepeatVector(10)(enc_layer)

output = HiddenStateLSTM2(
    1024,
    return_sequences=True
)([repeat, hidden[0], hidden[1]])


#repeat = RepeatVector(48)(enc_derp)
#print(repeat)
#dec_layer, _, _ = HiddenStateLSTM2(1024, dropout_W=0.5, dropout_U=0.5, return_sequences=True)([repeat, hidden[0], hidden[1]])
#dec_layer = TimeDistributed(Dense(19))(dec_layer)

#dec_output = Dropout(0.2)(dec_layer)
#dec_output = Activation('softmax', name='decoder_output')(dec_output)

model = Model(input=enc_input, output=output)

sgd = SGD(lr=1., decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

plot(model, to_file='test1.png', show_shapes=True)
