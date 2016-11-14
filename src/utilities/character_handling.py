#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import Config


class CharacterHandling:

    CHARACTERS = None

    def __init__(self):
        pass

    @staticmethod
    def fetch_characters():
        CharacterHandling.CHARACTERS = Config.get('general.characters')

    @staticmethod
    def char_to_index(char):
        if CharacterHandling.CHARACTERS is None:
            CharacterHandling.fetch_characters()

        for ii in range(len(CharacterHandling.CHARACTERS)):
            if CharacterHandling.CHARACTERS[ii] == char:
                return ii
        return -1
