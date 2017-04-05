# -*- coding: utf-8 -*-

from keras.layers import LSTM, Activation, Dense, Embedding, InputLayer, TimeDistributed
from keras.models import Sequential
from keras.regularizers import ActivityRegularizer, WeightRegularizer
from keras.utils.visualize_util import plot

from rorschach.prediction.keras.networks import BaseKerasPredictor
from rorschach.prediction.keras.tools import DimCalculator
from rorschach.utilities import Config


class LSTMEmbeddingSimplePredictor(BaseKerasPredictor):

    def prepare(self):
        super().prepare()

        input_width = DimCalculator.width(self.training_images_transformed)
        output_depth = DimCalculator.depth()

        self.model = Sequential()

        self.model.add(
            InputLayer(
                batch_input_shape=(
                    Config.get('predicting.batch_size'),
                    input_width
                )
            )
        )

        self.model.add(
            Embedding(
                self.data['voc_size_input'] + 1,
                1024,
                mask_zero=True
            )
        )

        self.model.add(
            LSTM(
                output_dim=256,
                return_sequences=True,
                W_regularizer=WeightRegularizer(l1=0.01, l2=0.01),
                b_regularizer=ActivityRegularizer(l1=0.01, l2=0.01),
            )
        )

        self.model.add(TimeDistributed(Dense(output_dim=output_depth)))

        self.model.add(Activation('softmax'))

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=[
                'categorical_accuracy',
            ]
        )

        self.model.summary()

        # Bootstrap our model to the bridge
        self.callback.model = self.model

        plot(self.model, to_file='model_rnn.png', show_shapes=True)