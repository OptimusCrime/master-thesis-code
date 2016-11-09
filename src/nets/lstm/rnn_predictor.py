#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from nets.base import BasePredictor
from utilities import Config, CharacterHandling


class RNNPredictor(BasePredictor):

    def __init__(self):
        super().__init__()

        self.widest = None

        # Storing the transformed data
        self.training_images_transformed = None
        self.training_labels_transformed = None
        self.phrase_transformed = None
