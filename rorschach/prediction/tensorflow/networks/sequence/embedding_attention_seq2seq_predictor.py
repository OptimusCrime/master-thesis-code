# -*- coding: utf-8 -*-

from rorschach.prediction.tensorflow.networks.sequence import AbstractSeq2SeqPredictor
from rorschach.prediction.tensorflow.layers import AttentionSeq2Seq


class EmbeddingAttentionSeq2SeqPredictor(AbstractSeq2SeqPredictor):

    def build_tf_model(self):
        self.model = AttentionSeq2Seq(
            xseq_len=self.training_images_transformed.shape[-1],
            yseq_len=self.training_labels_transformed.shape[-1],
            xvocab_size=self.data['voc_size_input'],
            yvocab_size=self.data['voc_size_labels'],
            emb_dim=1024,
            num_layers=3
        )
