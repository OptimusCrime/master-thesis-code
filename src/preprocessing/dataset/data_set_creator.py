#!/usr/bin/env python
# -*- coding: utf-8 -*-

from preprocessing.abstracts import AbstractImageSet
from preprocessing.handlers import TextCreator
from utilities import Config, Filesystem, pickle_data


class DataSetCreator(AbstractImageSet):

    def create(self):
        self.calculate_constraints()
        self.apply_constraints()

    def calculate_constraints(self):
        for character in Config.get('characters'):
            character_image = TextCreator.write(character)

            self.constraint_handler.transform_and_apply(character_image, character)

            self.images.append({
                'character': character,
                'object': character_image
            })

        self.constraint_handler.save()

    def apply_constraints(self):
        for i in range(len(self.images)):
            self.images[i]['object'] = self.images[i]['object'].crop(self.constraint_handler.get_constraints())

    def transform_and_dump(self):
        # Run parent method. Transform data
        transformed = super(DataSetCreator, self).transform_and_dump()

        # Store data for later use
        pickle_data(transformed, Filesystem.get_root_path('data/data_set.pickl'))
