# -*- coding: utf-8 -*-

import gc
import os
import json
from random import randrange

from rorschach.preprocessing.creators import AbstractCreator
from rorschach.preprocessing.handlers import TextCreator
from rorschach.utilities import Config, unpickle_data, JsonConfigEncoder


class TermCreator(AbstractCreator):

    def __init__(self, type):
        super().__init__(type)

    def random_font(self):
        if len(self.fonts) == 0:
            return self.fonts[0]

        return self.fonts[randrange(0, len(self.fonts))]

    @staticmethod
    def font_path_to_name(path):
        return path.split(os.sep)[-1].split('.')[0]

    def create_sets(self):
        contents = []

        # Storing information about fonts
        multiple_fonts = len(self.fonts) > 0

        if multiple_fonts:
            for font in self.fonts:
                Config.set('multiple_fonts_' + TermCreator.font_path_to_name(font), 0)

        data_set_size = len(self.terms)
        for i in range(data_set_size):
            if (i + 1) % Config.get('logging.batch_reporting') == 0:
                self.log.info('Constructing image %s/%s.', i + 1, data_set_size)

            font = self.random_font()
            if multiple_fonts:
                Config.inc('multiple_fonts_' + TermCreator.font_path_to_name(font), 0)

            phrase_arr = TextCreator.write(self.terms[i], self.set_type_keyword, font)

            contents.append({
                'text': self.terms[i],
                'matrix': phrase_arr
            })

            if i != 0 and i % 1000 == 0:
                gc.collect()

        if multiple_fonts:
            data = {}
            for font in self.fonts:
                font_name = TermCreator.font_path_to_name(font)
                data[font_name] = Config.get('multiple_fonts_' + font_name)

            with open(Config.get_path('path.data', 'fonts_' + self.set_type_keyword + '.json'), 'w') as outfile:
                json.dump(data, outfile, cls=JsonConfigEncoder)

        return contents

    def apply_constraints(self):
        constraints_file = Config.get_path('path.data', 'constraints.pickl')

        # We can not apply constraints unless we have the pickle file
        assert os.path.isfile(constraints_file) is True

        # Load the constraints for upper/lower lines
        char_constraints = unpickle_data(constraints_file)

        data_set_size = len(self.contents)
        for i in range(data_set_size):
            if (i + 1) % Config.get('logging.batch_reporting') == 0:
                self.log.info('Applying constraints for image %s/%s.', i + 1, data_set_size)

            # Calculate constraints for THIS unique term
            self.constraint_handler.reset()
            self.constraint_handler.calculate(self.contents[i]['matrix'])

            self.contents[i]['matrix'] = self.contents[i]['matrix'][
                                         char_constraints['top']:char_constraints['bottom'],
                                         self.constraint_handler.constraints['left']:
                                         self.constraint_handler.constraints['right']]
