#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorchach.preprocessing.creators import DataSetCreator


class TestDerp:

    def test_one(self):
        data_set_creator = DataSetCreator()

        assert data_set_creator is not None
