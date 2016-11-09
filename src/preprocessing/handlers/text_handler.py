#!/usr/bin/env python
# -*- coding: utf-8 -*-

# noinspection PyPackageRequirements
from PIL import Image
# noinspection PyPackageRequirements
from PIL import ImageDraw
# noinspection PyPackageRequirements
from PIL import ImageFont

from utilities import Config, Filesystem


class TextCreator:

    def __init__(self):
        pass

    @staticmethod
    def write(text):
        im = Image.new('1', Config.get('canvas_size'), 1)
        draw = ImageDraw.Draw(im)

        font_object = ImageFont.truetype(Config.get('text-font'), Config.get('text-size'))
        draw.text((0, 0), text, font=font_object, fill=0)

        if Config.get('preprocessing-save'):
            Filesystem.create('data/raw')
            im.save(Filesystem.get_root_path('data/raw/' + text + '.png'))

        return im
