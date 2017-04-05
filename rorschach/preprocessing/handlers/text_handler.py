# -*- coding: utf-8 -*-

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rorschach.utilities import Config, Filesystem


class TextCreator:

    def __init__(self):
        pass

    @staticmethod
    def write(text, data_set):
        im = Image.new('1', Config.get('preprocessing.canvas.size'), 1)
        draw = ImageDraw.Draw(im)

        font_object = ImageFont.truetype(Config.get('preprocessing.text.font'),
                                         Config.get('preprocessing.text.size'))
        draw.text((0, 0), text, font=font_object, fill=0)

        if Config.get('preprocessing.save.canvas'):
            im.save(
                Filesystem.save(
                    os.path.join(
                        Config.get('path.image'),
                        'canvas',
                        data_set
                    ),
                    text + '.png'
                )
            )

        arr = np.fromiter(list(im.getdata()), dtype="bool").reshape((im.height,
                                                                    im.width))

        im.close()

        del im
        del draw
        del font_object

        return arr
