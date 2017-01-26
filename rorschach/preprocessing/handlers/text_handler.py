#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image, ImageDraw, ImageFont

from rorschach.utilities import Config, Filesystem


class TextCreator:

    def __init__(self):
        pass

    @staticmethod
    def write(text):
        im = Image.new('1', Config.get('preprocessing.canvas.size'), 1)
        draw = ImageDraw.Draw(im)

        font_object = ImageFont.truetype(Config.get('preprocessing.text.font'),
                                         Config.get('preprocessing.text.size'))
        draw.text((0, 0), text, font=font_object, fill=0)

        if Config.get('preprocessing.save.raw'):
            Filesystem.create('data/raw')
            im.save(Filesystem.get_root_path('data/raw/' + text + '.png'))

        return im
