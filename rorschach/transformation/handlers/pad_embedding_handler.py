#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from rorschach.preprocessing.handlers import PadHandler


class PadEmbeddingHandler(PadHandler):

    def calculate_widest(self):
        for set_list_obj in self.set_list:
            for content in set_list_obj['data']:
                matrix_length = len(content['embedding'])

                if self.widest is None or matrix_length > self.widest:
                    self.widest = matrix_length

    def add_padding(self):
        for i in range(len(self.set_list)):
            for j in range(len(self.set_list[i]['data'])):
                # In the regular pad, we just combine two numpy arrays. Here we know that the data is a vector, so it
                # is just as easy to loop over the content. Perhaps a bit slower, but whatever. We pad the embedding
                # array with zeroes. We are going to expand and construct the label matrix when we save the set, so we
                # don't care about that here.
                old_matrix = self.set_list[i]['data'][j]['embedding']

                new_matrix = np.full((self.widest), '0', dtype=(np.str, 35))
                for v in range(len(old_matrix)):
                    new_matrix[v] = old_matrix[v]

                self.set_list[i]['data'][j]['embedding'] = new_matrix
