#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from rorschach.utilities import Config, unpickle_data


class PlotVector:

    COLORS = ['r', 'b', 'g']

    def __init__(self):
        self.mode = None
        while True:
            user_mode = input('Chose mode, [p] for PCA, [t] for TSNE: ')
            if user_mode in ['p', 't']:
                self.mode = user_mode
                break

            print('Invalid mode, please try again')
            print('')

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
                'word': key,
                'color': PlotVector.COLORS[len(words)],
                'vectors': []
            })

            for i in range(min(len(value), 4000)):
                temp_vector = []
                for inner_vector in value[i]['state']:
                    temp_vector.extend(inner_vector)
                vectors_raw.append(temp_vector)
                words[-1]['vectors'].append(vector_index)
                vector_index += 1

        x = np.array(vectors_raw, dtype=np.float32)

        if self.mode == 'p':
            model = PCA(n_components=2)
            model.fit(x)
            transformed = model.transform(x)
        else:
            model = TSNE(n_components=2, perplexity=100, n_iter=10000, verbose=True)
            transformed = model.fit_transform(x)

        colors = []
        for word in words:
            for _ in word['vectors']:
                colors.append(word['color'])

        scatter_x = []
        scatter_y = []
        for el in transformed:
            scatter_x.append(el[0])
            scatter_y.append(el[1])

        fig, ax = plt.subplots()
        if self.mode == 'p':
            ax.scatter(scatter_x, scatter_y, c=colors, alpha=1)
        else:
            sizes = [100] * len(scatter_x)
            ax.scatter(scatter_x, scatter_y, c=colors, s=sizes, alpha=1)
        ax.grid(True)

        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])

        fig.tight_layout()

        fig.savefig(Config.get_path('path.output', 'context.png', fragment=Config.get('uid')))


if __name__ == '__main__':
    plot_vector = PlotVector()
    plot_vector.run()
