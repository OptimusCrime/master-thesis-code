#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.main import Tester, Trainer
from rorschach.utilities import Config


def main():
    if Config.get('general.mode') == 'train' or Config.get('general.mode') == 'both':
        Trainer.run()

    if Config.get('general.mode') == 'test' or Config.get('general.mode') == 'both':
        Tester.run()

    if Config.get('general.mode') in ['test', 'train', 'both']:
        return

    raise Exception('Unknown mode. Should be either train, test or both.')


if __name__ == '__main__':
    main()
