#!/usr/bin/env python
# -*- coding: utf-8 -*-

# noinspection PyPackageRequirements
from PIL import Image
# noinspection PyPackageRequirements
from PIL import ImageDraw
# noinspection PyPackageRequirements
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
