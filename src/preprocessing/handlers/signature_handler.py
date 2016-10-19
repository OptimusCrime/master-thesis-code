#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utilities import Config


class SignatureHandler:

    def __init__(self):
        pass

    @staticmethod
    def apply(images):
        signature_position = Config.get('signature-position')
        signature_height = Config.get('signature-height')

        for i in range(len(images)):
            images[i]['object'] = images[i]['object'].crop((0, signature_position, images[i]['object'].width,
                                                            signature_position + signature_height))
