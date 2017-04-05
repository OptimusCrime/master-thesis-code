# -*- coding: utf-8 -*-

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler


class ConcatenateBinaryDataHandler(BaseHandler):

    def obj_handler(self, obj):
        expression = []
        current_offset = 0
        current_type = None
        current_length = 0

        matrix = obj[DataSetTypes.IMAGES]['matrix'][0]

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

        obj[DataSetTypes.IMAGES]['concatenated_binary'] = expression

        return obj
