# -*- coding: utf-8 -*-

from rorschach.prediction.common import TransformationHandlerNoiseApplier
from rorschach.prediction.tensorflow.layers import RNNSeq2Seq
from rorschach.prediction.tensorflow.networks.sequence import AbstractSeq2SeqPredictor
from rorschach.utilities import Config


class ContextVectorPredictor(AbstractSeq2SeqPredictor):

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

            # Rearrange the values to work with the context mapping
            'transformation.handlers.input.RearrangeContextVectorSequenceValuesHandler',

            # Translate the label text to corresponding integer ids (A -> 1, D -> 4, ...)
            'transformation.handlers.output.IntegerifyLabelHandler',

            # Swap inputs and labels for context mapping
            'transformation.handlers.finalize.SwapContextHandler'
        ]

        TransformationHandlerNoiseApplier.run(self.transformation_handlers)

    def build_tf_model(self):
        self.model = RNNSeq2Seq(
            self,
            xseq_len=self.training_images_transformed.shape[-1],
            yseq_len=self.training_labels_transformed.shape[-1],
            xvocab_size=Config.get('dataset.voc_size_input'),
            yvocab_size=Config.get('dataset.voc_size_output'),
            emb_dim=Config.get('predicting.embedding-dim'),
            num_layers=Config.get('predicting.rnn-group-depth')
        )
