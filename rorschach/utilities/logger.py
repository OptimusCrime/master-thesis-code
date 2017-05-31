# -*- coding: utf-8 -*-

import logging
import os
from logging.config import dictConfig

from rorschach.utilities import Config, Filesystem, UidGenerator


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

    INIT = False

    def __init__(self):
        pass

    @staticmethod
    def load(name, logger_type=DEFAULT):
        if not LoggerWrapper.INIT:
            UidGenerator.run()
            LoggerWrapper.INIT = True

        # This may be the ugliest hack ever. Because static variable are evaluated on initialization
        # we instead add this logic here, which is not ran before the first call to this function. We can then avoid
        # that our system creates a new directory for every uid seed.
        LoggerWrapper.LOG_CONFIGS[logger_type]['handlers']['file']['filename'] = \
            Config.get_path('path.output', 'run.log', fragment=Config.get('uid'))

        directory = Config.get_path('path.output', Config.get('uid'))
        if not os.path.exists(directory):
            Filesystem.create(directory, outside=True)

        dictConfig(LoggerWrapper.LOG_CONFIGS[logger_type])

        return logging.getLogger(name)
