#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
# noinspection PyPackageRequirements
from PIL import Image

from utilities import Filesystem


class VerificationSaver:

    def __init__(self):
        pass

    @staticmethod
    def save(directory, objects):
        Filesystem.create(directory)
        for obj in objects:
            VerificationSaver.save_image(obj['matrix'], obj['label'], obj['text'], directory)

    @staticmethod
    def save_image(matrix, labels, name, location):
        # Create new image
        new_img = Image.new('RGB', (matrix.shape[1], labels.shape[1] + 3), '#fff')
        new_img_pixels = new_img.load()

        # Draw the signature
        for x in range(len(matrix[0])):
            if matrix[0][x] == 0:
                new_img_pixels[x, 0] = (0, 0, 0)

        # Spacing
        for x in range(len(matrix[0])):
            new_img_pixels[x, 2] = (0, 0, 0)

        # Labels
        for x in range(len(labels)):
            y = int(np.argmax(labels[x]))
            if labels[x][y] == 0:
                continue

            new_img_pixels[x, y + 3] = (0, 0, 0)

        new_img.save(Filesystem.get_root_path(location + '/' + name + '.png'))
