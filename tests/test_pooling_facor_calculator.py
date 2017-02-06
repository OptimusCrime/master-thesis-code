#!/usr/bin/env python
# -*- coding: utf-8 -*-

from prediction.common.helpers import PoolingFactorCalculator

class TestPoolingFactorCalculator:

    def test_factor_first(self):
        factor = PoolingFactorCalculator.calc(32, 7)
        assert factor['factor'] == 4

    def test_factor_second(self):
        factor = PoolingFactorCalculator.calc(33, 7)
        assert factor['factor'] == 3

    def test_factor_third(self):
        factor = PoolingFactorCalculator.calc(10, 2)
        assert factor['factor'] == 5

    def test_factor_fouth(self):
        factor = PoolingFactorCalculator.calc(12, 2)
        assert factor['factor'] == 6

    def test_no_factor(self):
        factor = PoolingFactorCalculator.calc(11, 11)
        assert factor['factor'] == 1

    def test_adjust_first(self):
        factor = PoolingFactorCalculator.calc(12, 7, PoolingFactorCalculator.ADJUST)
        print(factor)
        assert factor['width_sequence'] == 14
        assert factor['factor'] == 2
