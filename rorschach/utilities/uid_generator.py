# -*- coding: utf-8 -*-

import datetime
import random
import string

from rorschach.utilities import Config


class UidGenerator:

    def __init__(self):
        pass

    @staticmethod
    def run():
        if Config.get('uid') is not None:
            return

        # Format yyyy-mm-dd hh:mm:ss.[timezone stuff]
        current_datetime = str(datetime.datetime.now())
        current_datetime_split = current_datetime.split(' ')

        uid_date = current_datetime_split[0] + '-' + current_datetime_split[1][0:8].replace(':', '-')
        uid_rand = ''.join(random.choice(string.ascii_lowercase) for x in range(6))
        Config.set('uid', uid_date + '-' + uid_rand)
