#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import gzip


def pickle_data(payload, filename):
    if filename is None:
        raise ValueError('filename must be defined (as full path)')
    if payload is None:
        raise ValueError('payload must be provided')

    with gzip.open(filename, 'wb') as f:
        pickle.dump(payload, f)


def unpickle_data(filename):
    if not filename:
        raise ValueError('filename must be defined (as full path)')

    try:
        with gzip.open(filename) as f:
            return pickle.load(f)
    except OSError:
        return None
