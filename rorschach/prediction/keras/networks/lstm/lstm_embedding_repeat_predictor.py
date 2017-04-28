# -*- coding: utf-8 -*-

import math

from keras.layers import LSTM, Activation, Dense, Embedding, Dropout, TimeDistributed, RepeatVector
from keras.models import Sequential
from keras.optimizers import Adam
from keras.initializers import RandomUniform

from rorschach.prediction.keras.networks import BaseKerasPredictor
from rorschach.prediction.keras.tools import DimCalculator
from rorschach.utilities import Config


class LSTMEmbeddingVectorPredictor(BaseKerasPredictor):

    LSTM_REPEAT_SIZE = 3

    def __init__(self):
        super().__init__()

        self.transformation_handlers = [
            # Initialize the labels
            'transformation.handlers.initializers.LabelInitializeHandler',

            # Translate individual bits to string representations
            'transformation.handlers.input.ConcatenateBinaryDataHandler',

            # Pad the input sequence to fit the longest sequence
            'transformation.handlers.input.PadHandler',

            # Translate text sequences into integers (1B -> -1, 6W -> 6, ...)
            'transformation.handlers.input.IntegerifyStringSequenceSpatialHandler',

            # Rearrange values from previous handler so the are all positive integers starting from 0
            'transformation.handlers.input.RearrangeSequenceValuesHandler',

            # Translate the label text to corresponding integer ids (A -> 1, D -> 4, ...)
            'transformation.handlers.output.IntegerifyLabelHandler',

            # Turn into one hot matrix
            'transformation.handlers.output.OneHotHandler',

            # Swap inputs and labels
            'transformation.handlers.finalize.SwapHandler'
        ]

    def build(self):
        super().build()

        self.model = Sequential()

        self.model.add(
            Embedding(
                Config.get('dataset.voc_size_input'),
                1024,
                input_length=self.dim_calculator.get(DimCalculator.INPUT_WIDTH),
                mask_zero=True,
                embeddings_initializer=RandomUniform(minval=math.sqrt(3), maxval=math.sqrt(3))
            )
        )

        # "Encoder"
        for i in range(LSTMEmbeddingVectorPredictor.LSTM_REPEAT_SIZE):
            # The last LSTM should not return the complete sequence
            self.model.add(
                LSTM(
                    1024,
                    return_sequences=(LSTMEmbeddingVectorPredictor.LSTM_REPEAT_SIZE - 1) != i,
                    recurrent_activation='sigmoid'
                )
            )

            self.model.add(Dropout(0.2))

        # Voodoo magic
        self.model.add(RepeatVector(self.dim_calculator.get(DimCalculator.LABELS_WIDTH)))

        # "Decoder"
        for _ in range(LSTMEmbeddingVectorPredictor.LSTM_REPEAT_SIZE):
            self.model.add(
                LSTM(
                    1024,
                    return_sequences=True,
                    recurrent_activation='sigmoid'
                )
            )

            self.model.add(Dropout(0.2))

        self.model.add(TimeDistributed(Dense(units=self.dim_calculator.get(DimCalculator.LABELS_DEPTH))))
        self.model.add(Activation('softmax'))

    def compile(self):
        optimizer = Adam(
            lr=Config.get('predicting.learning-rate')
        )

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=[
                'categorical_accuracy'
            ]
        )

        self.model.summary()

        # Bootstrap our model to the bridge
        self.callback.model = self.model
