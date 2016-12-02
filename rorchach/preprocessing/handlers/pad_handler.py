#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from preprocessing.handlers import BaseHandler


class PadHandler(BaseHandler):

    def __init__(self):
        super().__init__()
        self.widest = None

    def run(self):
        self.calculate_widest()
        self.add_padding()

    def calculate_widest(self):
        for set_list_obj in self.set_list:
            for content in set_list_obj['data']:
                matrix_length = len(content['matrix'][0])

                if self.widest is None or matrix_length > self.widest:
                    self.widest = matrix_length

    def add_padding(self):
        for i in range(len(self.set_list)):
            # Note: Not sure what this is supposed to do?
            # Check if this list has the correct dims
            # if len(self.set_list[i]['data'][0]['matrix'][0]) == self.widest:
            #     continue

            for j in range(len(self.set_list[i]['data'])):
                # Fetch the old matrix here
                old_matrix = self.set_list[i]['data'][j]['matrix']

                # NOTE:
                # np.concatenate merges two tables, BUT it does not overwrite the overlapping parts of the matrices.
                # Instead it expands it, so one matrix (1, 1) concatenated with (1, 1) would be (1, 2) with axis=1.
                # To hack this, I simply create a matrix that is the size of the widest minus the old width. When
                # concatenating this, it will return a matrix with the correct (widest) width.
                new_matrix = np.ones((old_matrix.shape[0], self.widest - old_matrix.shape[1]))
                self.set_list[i]['data'][j]['matrix'] = np.concatenate((old_matrix, new_matrix), axis=1)
