#!/usr/bin/env python
# -*- coding: utf-8 -*-

from common.abstracts import AbstractPredictor
from utilities import Config, CharacterHandling

import numpy as np
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Input
from keras.models import Model


class CNNPredictor(AbstractPredictor):

    def __init__(self):
        super().__init__()

        self.widest = None
        self.model = None

        self.training_images_transformed = None
        self.training_labels_transformed = None
        self.phrase_transformed = None

        print(Config.get('preprocessing-shift'))
        print(Config.get('preprocessing-shifxt'))
        print(Config.get('preprocessixng-shifxt'))
        print(Config.get('cnn-sizes'))

    def preprocess(self):
        self.calculate_widest()

        self.transform_data_set()
        self.transform_phrase()

        self.keras_setup()

    def calculate_widest(self):
        self.widest = len(self.phrase[0]['matrix'][0])

        # The narrowest input our network can handle is 40
        if self.widest < 40:
            self.widest = 40

    def transform_data_set(self):
        self.training_images_transformed = np.ones((len(self.data_set), self.widest))
        self.training_labels_transformed = np.zeros((len(self.data_set), len(Config.get('characters'))))

        for i in range(len(self.data_set)):
            matrix = self.data_set[i]['matrix'][0]
            np.put(self.training_images_transformed[i], np.arange(len(matrix)), matrix)

            # Set correct label
            char_index = CharacterHandling.char_to_index(self.data_set[i]['character'])
            self.training_labels_transformed[i][char_index] = 1

        # Because
        self.training_images_transformed = self.training_images_transformed.reshape((len(self.data_set),
                                                                                     self.widest, 1))

    def transform_phrase(self):
        self.phrase_transformed = self.phrase[0]['matrix'][0].reshape((1, 1, self.widest, 1))
        print(self.phrase_transformed)
        print(self.phrase_transformed.shape)

    def keras_setup(self):
        input_shape = (1, self.widest, 1)
        img_input = Input(shape=input_shape)

        x = Convolution2D(16, 1, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
        x = Convolution2D(16, 1, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
        x = MaxPooling2D((1, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Convolution2D(32, 1, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
        x = Convolution2D(32, 1, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
        x = MaxPooling2D((1, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Convolution2D(64, 1, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
        x = Convolution2D(64, 1, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
        x = Convolution2D(64, 1, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
        x = MaxPooling2D((1, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Convolution2D(128, 1, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
        x = Convolution2D(128, 1, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
        x = Convolution2D(128, 1, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
        x = MaxPooling2D((1, 1), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Convolution2D(128, 1, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
        x = Convolution2D(128, 1, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
        x = Convolution2D(128, 1, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
        x = MaxPooling2D((1, 2), strides=(2, 2), name='block5_pool')(x)

        # Predict
        x = Flatten(name='flatten')(x)
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dense(1024, activation='relu', name='fc2')(x)
        x = Dense(512, activation='relu', name='fc3')(x)
        x = Dense(len(Config.get('characters')), activation='softmax', name='predictions')(x)

        # Create model
        self.model = Model(img_input, x)

    def predict(self):
        self.predictions = self.model.predict(self.phrase_transformed)
        print(self.predictions)
