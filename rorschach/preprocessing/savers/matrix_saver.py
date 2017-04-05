# -*- coding: utf-8 -*-

import os

from PIL import Image

from rorschach.utilities import Config, Filesystem


class MatrixSaver:

    FONT_PATH_TO_NAME = {}

    def __init__(self):
        pass

    @staticmethod
    def save(directory, objects):
        Filesystem.create(os.path.join(Config.get('path.image'), directory), outside=True)

        for obj in objects:
            font = None
            if 'font' in obj:
                font = obj['font']
            MatrixSaver.save_image(obj['matrix'], obj['text'], directory, font)

    @staticmethod
    def font_name_cleaning(font):
        if font is None:
            return ''

        if len(Config.get('preprocessing.text.fonts')) == 1:
            return ''

        if font in MatrixSaver.FONT_PATH_TO_NAME:
            return MatrixSaver.FONT_PATH_TO_NAME[font]

        font_clean = font.split(os.sep)[-1]
        filetype_clean = font_clean.split('.')[0]

        MatrixSaver.FONT_PATH_TO_NAME[font] = '_' + filetype_clean

        return MatrixSaver.font_name_cleaning(font)


    @staticmethod
    def save_image(matrix, name, directory, font=None):
        # Create new image
        new_img = Image.new('RGB', matrix.shape[::-1], '#fff')
        new_img_pixels = new_img.load()

        # Color the letter in gray
        for y in range(len(matrix)):
            for x in range(len(matrix[0])):
                if matrix[y][x] == 0:
                    new_img_pixels[x, y] = (0, 0, 0)

        # Clean the font name
        file_name = name + MatrixSaver.font_name_cleaning(font)

        # Save the image
        new_img.save(Filesystem.save(os.path.join(Config.get('path.image'), directory), file_name + '.png'))
