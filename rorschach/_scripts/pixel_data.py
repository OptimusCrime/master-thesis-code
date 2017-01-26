#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image

file = input('File: ')

im = Image.open(file)
rgb_im = im.convert('RGB')

data = 0
for i in range(im.height):
    for j in range(im.width):
        color = rgb_im.getpixel((j, i))
        if color == (0, 0, 0):
            data += 1

print('Black pixels found ', data)
