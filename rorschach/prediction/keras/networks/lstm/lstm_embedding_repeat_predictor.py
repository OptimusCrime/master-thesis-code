# -*- coding: utf-8 -*-

from keras.layers import LSTM, Activation, Dense, Dropout, Embedding, TimeDistributed, RepeatVector
from keras.models import Sequential
from keras.regularizers import ActivityRegularizer, WeightRegularizer
from keras.utils.visualize_util import plot

from rorschach.prediction.keras.networks import BaseKerasPredictor
from rorschach.prediction.keras.tools import DimCalculator
from rorschach.utilities import Config


class LSTMEmbeddingVectorPredictor(BaseKerasPredictor):

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

    def prepare(self):
        super().prepare()

        self.model = Sequential()

        self.model.add(
            Embedding(
                Config.get('dataset.voc_size_input'),
                1024,
                input_length=self.dim_calculator.get(DimCalculator.INPUT_WIDTH),
                mask_zero=True,
            )
        )

        self.model.add(
            LSTM(
                output_dim=256
            )
        )

        self.model.add(RepeatVector(self.dim_calculator.get(DimCalculator.LABELS_WIDTH)))

        for _ in range(3):
            self.model.add(
                LSTM(
                    output_dim=256,
                    return_sequences=True
                )
            )

        # self.model.add(Dropout(0.1))

        self.model.add(TimeDistributed(Dense(output_dim=self.dim_calculator.get(DimCalculator.LABELS_DEPTH))))

        self.model.add(Activation('softmax'))

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=[
                'categorical_crossentropy',
                'categorical_accuracy'
            ]
        )

        self.model.summary()

        # Bootstrap our model to the bridge
        self.callback.model = self.model

        # Plot
        plot(
            self.model,
            to_file=Config.get_path('path.output', 'model.png', fragment=Config.get('uid')),
            show_shapes=True
        )
