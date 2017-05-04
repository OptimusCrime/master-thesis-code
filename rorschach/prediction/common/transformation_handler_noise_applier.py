# -*- coding: utf-8 -*-

from rorschach.utilities import Config


class TransformationHandlerNoiseApplier:

    def __init__(self):
        pass

    @staticmethod
    def run(handlers):
        if Config.get('transformation.noise-random-factor') in [None, 0]:
            return handlers

        handlers.insert(1, 'transformation.handlers.input.NoiseHandler')
