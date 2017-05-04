# -*- coding: utf-8 -*-


class BaseHandler:

    def __init__(self):
        self.input_lists = None

    def prepare(self):
        pass

    def run(self, input_lists):
        self.input_lists = input_lists

        for key in input_lists:
            adjusted_list = self.list_handler(input_lists[key], key)

            if adjusted_list is not None:
                input_lists[key] = adjusted_list

        self.finish()

        return input_lists

    def list_handler(self, input_list, key):
        if not input_list:
            return

        for i in range(len(input_list)):
            input_list[i] = self.obj_handler(input_list[i])

        return input_list

    def obj_handler(self, obj):
        return obj

    def finish(self):
        pass
