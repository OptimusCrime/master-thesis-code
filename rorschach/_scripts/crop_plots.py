#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob

from PIL import Image

from rorschach.utilities import Config


class CropPlots:

    def __init__(self, uid='plots'):
        Config.set('uid', uid)

    def run(self):
        glob_expression = Config.get_path('path.output', Config.get('uid')) + os.sep + '**' + os.sep + '*.png'
        files = glob.glob(glob_expression, recursive=True)

        for file in files:
            filepath_split = file.split(os.sep)
            filename_split = filepath_split[-1].split('.')
            if filename_split[0].endswith('_crop'):
                continue

            print('From = ')
            print(file)
            print('To = ')
            new_filepath = os.sep.join(filepath_split[:len(filepath_split) - 1]) + os.sep + filename_split[0] + '_crop.' + filename_split[1]
            print(new_filepath)
            print('')
            print('')

            #break

            #img = Image.open(file)
            #img2 = img.crop((0, 0, 100, 100))
            #img2.save("img2.jpg")


if __name__ == '__main__':
    crop_plots = CropPlots()
    crop_plots.run()
