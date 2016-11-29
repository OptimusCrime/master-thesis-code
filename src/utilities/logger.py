#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from logging.config import dictConfig


class LoggerWrapper:

    def __init__(self):
        pass

    @staticmethod
    def load(name):
        dictConfig(LOG_CONFIG)

        return logging.getLogger(name)


LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'default',
            'filename': os.path.join(os.path.dirname(__name__), '../', 'logs', 'debug.log'),
            'maxBytes': 1024 * 1024 * 10,
            'backupCount': 1
        }
    },
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s %(name)s.%(funcName)s:%(lineno)d %(message)s'
        }
    },
    'loggers': {
        '': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}