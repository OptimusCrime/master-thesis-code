#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.transformation.handlers import BaseHandler


class ConcatenateBinaryDataHandler(BaseHandler):

    def __init__(self):
        super().__init__()

    def run(self, input_lists):
        super().run(input_lists)

        return input_lists

    def list_handler(self, input_list, key):
        for obj in input_list['images']:
            self.obj_handler(obj, None)

    def obj_handler(self, ipt, label):
        expression = []
        current_offset = 0
        current_type = None
        current_length = 0

        matrix = ipt['matrix'][0]

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

        ipt['concatenated_binary'] = expression
