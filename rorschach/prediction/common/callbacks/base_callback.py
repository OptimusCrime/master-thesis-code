# -*- coding: utf-8 -*-


class BaseCallback:

    def __init__(self):
        self.data = None
        self.flags = None
        self.information = None

    def run(self):
        raise Exception('Missing callback run implementation')
