# -*- coding: utf-8 -*-

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rorschach.common import DataSetTypes
from rorschach.preprocessing.savers import MatrixSaver
from rorschach.utilities import Config, Filesystem


class TextCreator:

    def __init__(self):
        pass

    @staticmethod
    def write(text, data_set, font):
        im = Image.new('1', Config.get('preprocessing.canvas.size'), 1)
        draw = ImageDraw.Draw(im)

        font_object = ImageFont.truetype(
            font,
            Config.get('preprocessing.text.size')
        )

        draw.text((0, 0), text, font=font_object, fill=0)

        if Config.get('preprocessing.save.canvas'):
            file_name = text
            if data_set == DataSetTypes.type_to_keyword(DataSetTypes.LETTER_SET) and \
                    len(Config.get('preprocessing.text.fonts')) > 1:
                file_name += MatrixSaver.font_name_cleaning(font)

            im.save(
                Filesystem.save(
                    os.path.join(
                        Config.get('path.image'),
                        'canvas',
                        data_set
                    ),
                    file_name + '.png'
                )
            )

        arr = np.fromiter(list(im.getdata()), dtype="bool").reshape((im.height,
                                                                    im.width))

        im.close()

        del im
        del draw
        del font_object

        return arr
