# -*- coding: utf-8 -*-

import time


class TimeParse:

    def __init__(self):
        pass

    @staticmethod
    def parse(start):
        seconds = round(time.time() - start)
        minutes = 0

        if seconds > 60:
            minutes = seconds // 60
            seconds = seconds - (minutes * 60)

        output = []
        if minutes > 0:
            output.append('{:,} minute'.format(minutes) + ('s' if minutes > 1 else ''))
        if seconds > 0:
            output.append('{} second'.format(seconds) + ('s' if seconds > 1 else ''))

        if len(output) > 1:
            return ' and '.join(output)
        else:
            return output[0]
