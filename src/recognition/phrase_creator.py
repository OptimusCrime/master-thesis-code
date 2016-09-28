#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.dataset.constraint_creator import ConstraintCreator
from src.handlers.constraint_handler import ConstraintHandler
from src.handlers.text_handler import TextCreator
from src.utilities.config import Config
from src.utilities.filesystem import Filesystem
from src.utilities.pickler import unpickle_data

import os


class PhraseCreator:

    def __init__(self):
        self.image = None
        self.constraint_handler = ConstraintHandler()

    def create(self):
        # Check if we have any constraints defined
        if Config.get('force') or not os.path.exists(Filesystem.get_root_path('data/constraints.pickl')):
            ConstraintCreator.create()

        self.create_image()
        self.calculate_crop()
        self.apply_crop()

    def create_image(self):
        # Create the image
        self.image = TextCreator.write(Config.get('phrase'))

    def calculate_crop(self):
        # Load the constraints for upper/lower lines
        data_set_constraints = unpickle_data(Filesystem.get_root_path('dump/data/constraints.pickl'))

        # Calculate the constraints for the text (left/right)
        self.constraint_handler.transform_and_apply(self.image, None)

        # Combine the constraint from the data set
        self.constraint_handler.combine(data_set_constraints)

    def apply_crop(self):
        self.image = self.image.crop(self.constraint_handler.get_constraints())

        # Aaaaand save the image
        self.image.save(Filesystem.get_root_path('pls.png'))

if __name__ == '__main__':
    phrase_creator = PhraseCreator()
    phrase_creator.create()
