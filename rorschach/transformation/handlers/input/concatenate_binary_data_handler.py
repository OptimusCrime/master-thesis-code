#!/usr/bin/env python
# -*- coding: utf-8 -*-


class ConcatenateBinaryDataHandler:

    def __init__(self):
        pass

    @staticmethod
    def run(input_lists):
        for data_list in input_lists:
            for obj in data_list['set']:
                ConcatenateBinaryDataHandler.concatenate_binary(obj)

    @staticmethod
    def concatenate_binary(obj):
        expression = []
        current_offset = 0
        current_type = None
        current_length = 0

        matrix = obj['matrix'][0]

        for i in range(len(matrix)):
            if current_type is None:
                current_type = matrix[current_offset]
                current_offset += 1
                current_length = 1
                continue

            if matrix[current_offset] != current_type:
                expression.append(str(current_length) + ('B' if current_type == 0 else 'W'))
                current_length = 0
                current_type = matrix[current_offset]

            current_length += 1
            current_offset += 1

        expression.append(str(current_length) + ('B' if current_type == 0 else 'W'))

        obj['concatenated_binary'] = expression
