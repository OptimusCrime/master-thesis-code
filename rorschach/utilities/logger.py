#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from logging.config import dictConfig


class LoggerWrapper:

    DEFAULT = 'default'
    SIMPLE = 'simple'

    def __init__(self):
        pass

    @staticmethod
    def load(name, logger_type=DEFAULT):
        dictConfig(LOG_CONFIGS[logger_type])

        return logging.getLogger(name)


LOG_CONFIGS = {
    LoggerWrapper.DEFAULT: {
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
    },
    LoggerWrapper.SIMPLE: {
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
                'format': '[%(asctime)s] %(levelname)s %(name)s %(message)s'
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
}
