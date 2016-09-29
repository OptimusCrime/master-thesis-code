#!/usr/bin/env python
# -*- coding: utf-8 -*-

from preprocessing.abstracts import AbstractImageSet
from preprocessing.handlers import TextCreator
from preprocessing.utilities import Config, Filesystem, pickle_data, unpickle_data


class PhraseCreator(AbstractImageSet):

    def create(self):
        self.create_image()
        self.calculate_crop()
        self.apply_crop()

    def create_image(self):
        phrase_image = TextCreator.write(Config.get('phrase'))

        self.images = [{
            'character': None,
            'object': phrase_image
        }]

    def calculate_crop(self):
        # Load the constraints for upper/lower lines
        data_set_constraints = unpickle_data(Filesystem.get_root_path('data/constraints.pickl'))

        # Calculate the constraints for the text (left/right)
        self.constraint_handler.transform_and_apply(self.images[0]['object'], None)

        # Combine the constraint from the data set
        self.constraint_handler.combine(data_set_constraints)

    def apply_crop(self):
        self.images[0]['object'] = self.images[0]['object'].crop(self.constraint_handler.get_constraints())

    def transform_and_dump(self):
        # Run parent method. Transform data
        transformed = super(PhraseCreator, self).transform_and_dump()

        # Store data for later use
        pickle_data(transformed, Filesystem.get_root_path('data/phrase.pickl'))


if __name__ == '__main__':
    phrase_creator = PhraseCreator()
    phrase_creator.create()
