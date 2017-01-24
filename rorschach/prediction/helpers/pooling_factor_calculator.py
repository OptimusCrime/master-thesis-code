#!/usr/bin/env python
# -*- coding: utf-8 -*-


class PoolingFactorCalculator:

    CALC = 0
    ADJUST = 1

    def __init__(self):
        pass

    @staticmethod
    def calc(ipt, output, mode=CALC):
        input_width, output_width = PoolingFactorCalculator.variable_input_value(ipt, output)

        if mode == PoolingFactorCalculator.CALC:
            return PoolingFactorCalculator.run(input_width, output_width)

        return PoolingFactorCalculator.adjust_input_calc(input_width, output_width)

    @staticmethod
    def adjust_input_calc(ipt, output):
        intermediate_results = []
        for i in range(4):
            results = PoolingFactorCalculator.run(ipt + i, output)
            intermediate_results.append(results)

        best_result = None
        best_diff = None
        for result in intermediate_results:
            if best_result is None:
                best_result = result
                continue

            # Here we calculat the difference in the network structure
            current_diff = abs(best_result['width_label'] - result['width_label'])
            current_diff += abs(best_result['width_sequence'] - result['width_sequence'])

            # If the length of the labels is lower than our best label AND the overall difference from the original
            # input/ouput values are improved, we switch best results here
            if best_result['width_label'] > result['width_label'] and (best_diff is None or current_diff <= best_diff):
                best_result = result
                best_diff = current_diff

        return best_result

    @staticmethod
    def variable_input_value(ipt, output):
        input_width = ipt
        output_width = output

        if type(input_width) is not int:
            input_width = ipt.shape[1]

        if type(output_width) is not int:
            output_width = output.shape[2]

        return input_width, output_width

    @staticmethod
    def run(widest_input, longest_label):
        div_facors = set()
        div_list = PoolingFactorCalculator.list_divs(widest_input)
        for div in div_list:
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

        # Calculat the smallest pooling factor
        smallest_factor = widest_input // min(div_facors)

        # If the pooling factor is 12 we found no actually valid factors, instead we set factor to 1
        if smallest_factor == widest_input:
            smallest_factor = 1

        return {
            'factor': smallest_factor,
            'width_label': widest_input // smallest_factor,
            'width_sequence': widest_input
        }

    @staticmethod
    def list_divs(input_length):
        divs = []
        for i in range(2, input_length):
            divs.append(i)
        return divs
