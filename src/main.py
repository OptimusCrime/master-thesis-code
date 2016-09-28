#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.recognition.phrase_creator import PhraseCreator
from src.utilities.config import Config

def main():
    phrase_creator = PhraseCreator()
    phrase_creator.create()


if __name__ == '__main__':
    main()
