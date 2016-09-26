#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utilities.pickler import pickle_data

from PIL import Image

import numpy as np


class CreateDataSet:

    LETTER_SET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                  'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    def __init__(self):
        pass

    @staticmethod
    def create(signature_level=0, signature_height=1):
        images = CreateDataSet.load_images()

        data = CreateDataSet.adjust_images(images, signature_level, signature_height)

        CreateDataSet.dump_data_set(data)

    @staticmethod
    def load_images():
        images = []
        for letter in CreateDataSet.LETTER_SET:
            images.append({
                'object': Image.open('dump/letters/' + letter + '.png'),
                'letter': letter
            })

        return images

    @staticmethod
    def adjust_images(images, signature_level, signature_height):
        data = []
        for image in images:
            data.append({
                'matrix': CreateDataSet.adjust_image(image, signature_level, signature_height),
                'letter': image['letter']
            })

        return data

    @staticmethod
    def adjust_image(image, signature_level, signature_height):
        # First adjust according to signature
        signature_image = image['object'].crop((0, signature_level, image['object'].width,
                                                signature_level + signature_height))

        # Construct numpy array and reshape into matrix
        arr = np.fromiter(list(signature_image.getdata()), dtype="int").reshape((-1,
                                                                                 signature_image.width))

        # Get the leftmost and rightmost values, so we can cut away all the whitespace
        left_pad = None
        right_pad = None
        for line in arr:
            for x in range(len(line)):
                if line[x] == 0:
                    if left_pad is None or x < left_pad:
                        left_pad = x
                    if right_pad is None or x > right_pad:
                        right_pad = x

        # Slice! (I have no idea why I had to add + 1)
        return arr[::, left_pad:right_pad + 1]

    @staticmethod
    def dump_data_set(data):
        pickle_data(data, 'dump/data/signatures.pickl')


if __name__ == '__main__':
    CreateDataSet.create(17, 1)
