#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from predicting import Predicter
from preprocessing import Preprocessor
from utilities import Config, Filesystem

def main():
    # Do pre processing
    if Config.get('force') or not os.path.exists(Filesystem.get_root_path('data/data_set.pickl')):
        preprocessor = Preprocessor()
        preprocessor.run()

    # Do the predicting
    predicter = Predicter()
    predicter.run()


if __name__ == '__main__':
    main()
