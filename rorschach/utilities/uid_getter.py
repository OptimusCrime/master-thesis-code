# -*- coding: utf-8 -*-

import os

from rorschach.utilities import Config


class UidGetter:

    def __init__(self):
        pass

    @staticmethod
    def run():
        uid = input('Enter uid: ')

        if uid is None or len(uid) == 0:
            raise Exception('No valid uid provided')

        if not os.path.exists(Config.get_path('path.output', 'data.json', fragment=uid)):
            raise Exception('No valid uid provided')

        Config.set('uid', uid)
