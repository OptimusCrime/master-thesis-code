#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from logging.config import dictConfig

from rorschach.utilities import Config, UidGenerator

UidGenerator.run()


class LoggerWrapper:

    DEFAULT = 'default'
    SIMPLE = 'simple'

    LOG_CONFIGS = {
        DEFAULT: {
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
                    'filename': Config.get_path('path.output', 'debug.log', fragment=Config.get('uid')),
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
        SIMPLE: {
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
                    'filename': Config.get_path('path.output', 'run.log', fragment=Config.get('uid')),
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

    def __init__(self):
        pass

    @staticmethod
    def load(name, logger_type=DEFAULT):
        dictConfig(LoggerWrapper.LOG_CONFIGS[logger_type])

        return logging.getLogger(name)
