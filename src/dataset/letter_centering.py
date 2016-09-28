#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.handlers.constraint_handler import CenteringHandler
from src.handlers.text_handler import TextCreator
from src.utilities.filesystem import Filesystem

import numpy as np


class LetterCentering:

    # width, height
    IMAGE_SIZE = (50, 40)

    LETTER_SET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                  'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    def __init__(self, font):
        self.font = font
        self.images = []
        self.centering_handler = CenteringHandler()

    def center_font_letters(self):
        self.calculate_letter_constraints()

        self.center_images()

    def calculate_letter_constraints(self):
        for letter in LetterCentering.LETTER_SET:
            letter_image = TextCreator.write(letter, self.font, None, LetterCentering.IMAGE_SIZE)

            self.calculate_constraint_for_letter(letter, letter_image)

            self.images.append({
                'letter': letter,
                'object': letter_image
            })

    def calculate_constraint_for_letter(self, letter, img):
        arr = np.fromiter(list(img.getdata()), dtype="int").reshape((-1, img.width))

        # We just call our centering handler, because we need to do the same later
        self.centering_handler.apply(arr, letter)

    def center_images(self):
        for image in self.images:
            image['object'] = image['object'].crop(self.centering_handler.get_constraints())
            image['object'].save(Filesystem.get_root_path() + '/dump/letters/' + image['letter'] + '.png')


if __name__ == '__main__':
    letter_centering = LetterCentering(None)
    letter_centering.center_font_letters()
