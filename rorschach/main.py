#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.main import Tester, Trainer
from rorschach.utilities import Config


def main():
    if Config.get('general.mode') == 'train':
        return Trainer.run()

    if Config.get('general.mode') == 'test':
        return Tester.run()

    raise Exception('Unknown mode. Should be either train or test.')


if __name__ == '__main__':
    main()
