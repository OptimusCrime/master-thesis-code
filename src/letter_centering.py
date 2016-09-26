#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


class LetterCentering:

    # width, height
    IMAGE_SIZE = (50, 40)

    LETTER_SET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                  'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    IMAGE_CONSTRAINTS = {
        'left': {
            'value': None,
            'letter': None
        },
        'right': {
            'value': None,
            'letter': None
        },
        'top': {
            'value': None,
            'letter': None
        },
        'bottom': {
            'value': None,
            'letter': None
        }
    }

    def __init__(self):
        pass

    @staticmethod
    def center_font_letters(font):
        images = LetterCentering.calculate_letter_constraints(font)

        constraint = tuple([LetterCentering.IMAGE_CONSTRAINTS['left']['value'],
                            LetterCentering.IMAGE_CONSTRAINTS['top']['value'],
                            LetterCentering.IMAGE_CONSTRAINTS['right']['value'],
                            LetterCentering.IMAGE_CONSTRAINTS['bottom']['value']])

        LetterCentering.center_images(images, constraint)

    @staticmethod
    def calculate_letter_constraints(font):
        images = []

        for letter in LetterCentering.LETTER_SET:
            img = Image.new('1', LetterCentering.IMAGE_SIZE, 1)
            draw = ImageDraw.Draw(img)

            font_object = ImageFont.truetype('fonts/' + font + '.ttf', 35)
            draw.text((0, 0), letter, font=font_object, fill=0)

            LetterCentering.calculate_constraint_for_letter(letter, img)

            images.append({
                'letter': letter,
                'object': img
            })

        return images

    @staticmethod
    def calculate_constraint_for_letter(letter, img):
        arr = np.fromiter(list(img.getdata()), dtype="int").reshape((LetterCentering.IMAGE_SIZE[1],
                                                                     LetterCentering.IMAGE_SIZE[0]))

        # Loop the matrix on both axes, finding their very edge
        for i in range(LetterCentering.IMAGE_SIZE[1]):
            for j in range(LetterCentering.IMAGE_SIZE[0]):
                if arr[i][j] == 0:
                    # left
                    if LetterCentering.IMAGE_CONSTRAINTS['left']['value'] is None or \
                            LetterCentering.IMAGE_CONSTRAINTS['left']['value'] > j:
                        LetterCentering.IMAGE_CONSTRAINTS['left']['value'] = j
                        LetterCentering.IMAGE_CONSTRAINTS['left']['letter'] = letter

                    # right
                    if LetterCentering.IMAGE_CONSTRAINTS['right']['value'] is None or \
                            LetterCentering.IMAGE_CONSTRAINTS['right']['value'] < j:
                        LetterCentering.IMAGE_CONSTRAINTS['right']['value'] = j
                        LetterCentering.IMAGE_CONSTRAINTS['right']['letter'] = letter

                    # top
                    if LetterCentering.IMAGE_CONSTRAINTS['top']['value'] is None or \
                            LetterCentering.IMAGE_CONSTRAINTS['top']['value'] > i:
                        LetterCentering.IMAGE_CONSTRAINTS['top']['value'] = i
                        LetterCentering.IMAGE_CONSTRAINTS['top']['letter'] = letter

                    # bottom
                    if LetterCentering.IMAGE_CONSTRAINTS['bottom']['value'] is None or \
                            LetterCentering.IMAGE_CONSTRAINTS['bottom']['value'] < i:
                        LetterCentering.IMAGE_CONSTRAINTS['bottom']['value'] = i
                        LetterCentering.IMAGE_CONSTRAINTS['bottom']['letter'] = letter

    @staticmethod
    def center_images(images, constraints):
        for image in images:
            image['object'] = image['object'].crop(constraints)
            image['object'].save('dump/letters/' + image['letter'] + '.png')


if __name__ == '__main__':
    LetterCentering.center_font_letters('arial-mono')
