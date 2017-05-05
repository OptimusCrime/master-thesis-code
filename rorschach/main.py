#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.main import Tester, Trainer
from rorschach.utilities import Config


def main():
    if Config.get('general.mode') in ['train', 'both', 'continue']:
        Trainer.run()

    if Config.get('general.mode') in ['test', 'both', 'continue', 'predict']:
        Tester.run()

    if Config.get('general.mode') in ['test', 'train', 'both', 'continue', 'predict']:
        return

    raise Exception('Unknown mode. Should be either train, test or both.')


if __name__ == '__main__':
    main()
