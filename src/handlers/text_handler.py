#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.utilities.filesystem import Filesystem

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


class TextCreator:

    DEFAULT_FONT = 'arial-mono'
    DEFAULT_FONT_SIZE = 35
    DEFAULT_CANVAS_SIZE = (1000, 200)

    def __init__(self):
        pass

    @staticmethod
    def write(text, font, font_size, canvas_size):
        if font is None:
            font = TextCreator.DEFAULT_FONT
        if font_size is None:
            font_size = TextCreator.DEFAULT_FONT_SIZE
        if canvas_size is None:
            canvas_size = TextCreator.DEFAULT_CANVAS_SIZE

        img = Image.new('1', canvas_size, 1)
        draw = ImageDraw.Draw(img)

        font_object = ImageFont.truetype(Filesystem.get_root_path() + '/fonts/' + font + '.ttf', font_size)
        draw.text((0, 0), text, font=font_object, fill=0)

        return img
