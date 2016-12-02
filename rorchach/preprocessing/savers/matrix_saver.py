#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image

from rorchach.utilities import Filesystem


class MatrixSaver:

    def __init__(self):
        pass

    @staticmethod
    def save(directory, objects):
        Filesystem.create(directory)
        for obj in objects:
            MatrixSaver.save_image(obj['matrix'], obj['text'], directory)

    @staticmethod
    def save_image(matrix, name, location):
        # Create new image
        new_img = Image.new('RGB', matrix.shape[::-1], '#fff')
        new_img_pixels = new_img.load()

        # Color the letter in gray
        for y in range(len(matrix)):
            for x in range(len(matrix[0])):
                if matrix[y][x] == 0:
                    new_img_pixels[x, y] = (0, 0, 0)

        new_img.save(Filesystem.get_root_path(location + '/' + name + '.png'))
