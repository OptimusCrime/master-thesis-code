#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from rorschach.utilities import Config, unpickle_data


class PlotVector:

    COLORS = ['r', 'b', 'g']

    def __init__(self):
        pass

    def run(self):
        context_data_file = Config.get_path('path.output', 'context_data.pickl', fragment=Config.get('uid'))
        if not os.path.exists(context_data_file):
            raise Exception('File not found')

        context_data = unpickle_data(context_data_file)

        words = []
        vectors_raw = []
        vector_index = 0

        for key, value in context_data.items():
            words.append({
                'word':key,
                'color': PlotVector.COLORS[len(words)],
                'vectors': []
            })

            for i in range(len(value)):
                temp_vector = []
                for inner_vector in value[i]['state']:
                    temp_vector.extend(inner_vector)
                vectors_raw.append(temp_vector)
                words[-1]['vectors'].append(vector_index)
                vector_index += 1

        x = np.array(vectors_raw, dtype=np.float32)

        model = TSNE(n_components=2, random_state=0, method='exact')
        transformed = model.fit_transform(x)

        colors = []
        for word in words:
            for i in word['vectors']:
                colors.append(word['color'])

        scatter_x = []
        scatter_y = []
        for el in transformed:
            scatter_x.append(el[0])
            scatter_y.append(el[1])

        fig, ax = plt.subplots()
        ax.scatter(scatter_x, scatter_y, c=colors, alpha=0.5)
        ax.grid(True)
        fig.tight_layout()

        fig.savefig(Config.get_path('path.output', 'context.png', fragment=Config.get('uid')))


if __name__ == '__main__':
    plot_vector = PlotVector()
    plot_vector.run()
