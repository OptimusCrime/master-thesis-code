#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os

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

            new_file_path = os.sep.join(filepath_split[:len(filepath_split) - 1]) + os.sep
            new_file_name = filename_split[0] + '_crop.' + filename_split[1]

            img = Image.open(file)
            width = img.size[0]
            height = img.size[1]

            img2 = img.crop((88, 0, width - 174, height))
            img2.save(new_file_path + new_file_name)

            del img
            del img2

            print('Cropped plot from ')
            print(file)
            print('')


if __name__ == '__main__':
    crop_plots = CropPlots()
    crop_plots.run()
