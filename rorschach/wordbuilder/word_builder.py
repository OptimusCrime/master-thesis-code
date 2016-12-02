#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from rorschach.utilities import Config
from rorschach.wordbuilder.parser import ProbabilityParser


class WordBuilder:
    def __init__(self):
        self.probability_parser = None
        self.predictions = None

    def build(self):
        self.probability_parser = ProbabilityParser()
        self.probability_parser.run()

    def calculate(self, predictions):
        self.predictions = predictions

        self.normalize_predictions()
        self.calculate_first()
        self.calculate_transisitions()

        self.debug()

    def normalize_predictions(self):
        for prediction in self.predictions:
            prediction_matrix = prediction['values'][0]

            prediction_normalized = np.zeros((len(prediction_matrix)))
            prediction_sum = sum(prediction_matrix)
            for i in range(len(prediction_matrix)):
                prediction_normalized[i] = prediction_matrix[i] / prediction_sum

            # OVerwrite unnormalized prediction
            prediction['values'][0] = prediction_normalized

    def calculate_transisitions(self):
        previous_letter = None
        for i in range(len(self.predictions)):
            current_prediction = self.predictions[i]['values'][0]

            if previous_letter is None:
                previous_letter = np.argmax(current_prediction)
                continue

            from_probabilities = self.probability_parser.prob_trans[previous_letter]

            # Matrix multiplication
            self.predictions[i]['values'] = WordBuilder.matrix_mul(current_prediction, from_probabilities,
                                                                   lambda x, y: (x * 2) + (y * 0))

            sorted_probabilities = WordBuilder.sort_prabilities(self.predictions[i]['values'])

            previous_letter = sorted_probabilities[0]['index']

    def calculate_first(self):
        initial_prediction = self.predictions[0]['values']
        initial_probability = self.probability_parser.prob_initial

        # Matrix multiplication
        self.predictions[0]['values'] = WordBuilder.matrix_mul(initial_prediction[0],
                                                               initial_probability,
                                                               lambda x, y: (x * 1) + (y * 0))

    @staticmethod
    def sort_prabilities(matrix, keyword='value'):
        # Sort the predictions by probability
        ordered_prediction_probability = []
        for i in range(len(matrix)):
            ordered_prediction_probability.append({
                'value': matrix[i],
                'index': i,
                'letter': Config.get('characters')[i]
            })

        return sorted(ordered_prediction_probability, key=lambda x: x[keyword], reverse=True)

    @staticmethod
    def matrix_mul(initial, factor, formula):
        combined_matrix = np.empty_like(initial)
        for i in range(len(initial)):
            combined_matrix[i] = formula(initial[i], factor[i])
        return combined_matrix

    def debug(self):
        print('here')
        for i in range(len(self.predictions)):
            print(WordBuilder.sort_prabilities(self.predictions[i]['values']))
        print('done')
