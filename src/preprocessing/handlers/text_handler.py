#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from utilities import Config


class TextCreator:

    def __init__(self):
        pass

    @staticmethod
    def write(text):
        img = Image.new('1', Config.get('canvas_size'), 1)
        draw = ImageDraw.Draw(img)

        font_object = ImageFont.truetype(Config.get('text-font'), Config.get('text-size'))
        draw.text((0, 0), text, font=font_object, fill=0)

        return img
