#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.handlers.centering_handler import CenteringHandler
from src.handlers.text_handler import TextCreator
from src.utilities.filesystem import Filesystem

import numpy as np


class PhraseCreator:

    FONT_SIZE = 35

    def __init__(self, font, text):
        self.font = font
        self.text = text

    def create(self):
        # Create the image
        phrase_image = TextCreator.write(self.text, self.font, None, None)

        # Convert image into
        arr = np.fromiter(list(phrase_image.getdata()), dtype="int").reshape((-1, phrase_image.width))

        # Calculate centering
        centering_handler = CenteringHandler()
        centering_handler.apply(arr, None)

        # Crop image
        phrase_image = phrase_image.crop(centering_handler.get_constraints())

        # Aaaaand save the image
        phrase_image.save(Filesystem.get_root_path() + '/dump/phrase/derp.png')

if __name__ == '__main__':
    phrase_creator = PhraseCreator(None, 'HEI')
    phrase_creator.create()
