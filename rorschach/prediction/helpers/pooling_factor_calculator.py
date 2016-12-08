#!/usr/bin/env python
# -*- coding: utf-8 -*-


class PoolingFactorCalculator:

    def __init__(self):
        pass

    @staticmethod
    def calc(input, output):
        input_width = input.shape[1]
        output_width = output.shape[2]

        results = PoolingFactorCalculator.run(input_width, output_width)

        return results['factor']

    @staticmethod
    def run(widest_input, longest_label):
        div_facors = set()
        for div in [2, 3, 5]:
            valids = [widest_input]
            while True:
                calc = valids[-1] / float(div)
                if calc.is_integer():
                    valids.append(int(calc))
                    if calc < longest_label:
                        del valids[-1]
                        break

                div_facors.update(valids)

                break

        smallest_factor = min(div_facors)

        return {
            'factor': widest_input // smallest_factor,
            'label_width': smallest_factor
        }
