#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image

from rorchach.utilities import Config, Filesystem


class SignatureHandler:

    @staticmethod
    def save_signature(images):
        Filesystem.create('data/illustration')

        signature_position = Config.get('preprocessing.signature.position')
        signature_height = Config.get('preprocessing.signature.height')

        for i in range(len(images)):
            # Create matrix
            arr = np.fromiter(list(images[i]['object'].getdata()), dtype="int").reshape(
                (images[i]['object'].height, images[i]['object'].width))

            # Create new image
            new_img = Image.new('RGB', arr.shape[::-1], '#fff')
            new_img_pixels = new_img.load()

            # Color the letter in gray
            for y in range(len(arr)):
                for x in range(len(arr[0])):
                    if arr[y][x] == 0:
                        new_img_pixels[x, y] = (128, 128, 128)

            # Crop the signature image (for lazyness, we are cropping the image here, not just (re)using the array
            # we already have at hand. Should perhaps recode this
            cropped_signature_img = images[i]['object'].crop((0, signature_position, images[i]['object'].width,
                                                              signature_position + signature_height))

            signature_arr = np.fromiter(list(cropped_signature_img.getdata()), dtype="int").reshape(
                (cropped_signature_img.height, cropped_signature_img.width))

            for y in range(signature_position, signature_position + signature_height):
                for x in range(len(signature_arr[0])):
                    color = (211, 211, 211)
                    if signature_arr[signature_position - y][x] == 0:
                        color = (0, 0, 0)

                    new_img_pixels[x, y] = color

            new_img.save(Filesystem.get_root_path('data/illustration/' + images[i]['character'] + '.png'))
