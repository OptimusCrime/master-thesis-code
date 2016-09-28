#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.handlers.constraint_handler import ConstraintHandler
from src.handlers.text_handler import TextCreator
from src.utilities.config import Config


class ConstraintCreator:

    def __init__(self):
        pass

    @staticmethod
    def create():
        images = []
        constraint_handler = ConstraintHandler()

        for character in Config.get('characters'):
            character_image = TextCreator.write(character)

            constraint_handler.transform_and_apply(character_image, character)

            images.append({
                'character': character,
                'object': character_image
            })

        constraint_handler.save()
