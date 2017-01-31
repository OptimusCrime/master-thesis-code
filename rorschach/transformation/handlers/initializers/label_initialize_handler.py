# !/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.common import DataSetTypes
from rorschach.transformation.handlers import BaseHandler


class LabelInitializeHandler(BaseHandler):

    def list_handler(self, input_list, key):
        new_list = [{}] * len(input_list)

        for i in range(len(input_list)):
            new_list[i] = {
                DataSetTypes.IMAGES: input_list[i],
                DataSetTypes.LABELS: {
                    'text': input_list[i]['text']
                }
            }

        return new_list
