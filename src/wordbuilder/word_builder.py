#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from utilities import Config
from wordbuilder.parser import ProbabilityParser


class WordBuilder:
    def __init__(self):
        self.probability_parser = None
        self.predictions = None

    def build(self):
        self.probability_parser = ProbabilityParser()
        #self.probability_parser.run()

    def calculate(self, predictions):
        self.predictions = predictions

        #self.calculate_first()
        self.calculate_transisitions()

    def calculate_transisitions(self):
        previous_letter = None
        print(WordBuilder.sort_prabilities(self.predictions[0]['values'][0]))

        for i in range(len(self.predictions)):
            current_prediction = self.predictions[i]['values'][0]

            if previous_letter is None:
                previous_letter = np.argmax(current_prediction)
                continue

            #from_probabilities = self.probability_parser.prob_trans[previous_letter]

            # Matrix multiplication
            #prediction_probabilities = np.multiply(current_prediction, from_probabilities)

            ordered_prediction_prabilities = WordBuilder.sort_prabilities(current_prediction)
            print(ordered_prediction_prabilities)

            previous_letter = ordered_prediction_prabilities[0]['index']

    def calculate_first(self):
        initial_probability = self.probability_parser.prob_initial
        initial_prediction = self.predictions[0]['values']

        # Matrix multiplication
        prediction_probability = np.multiply(initial_probability, initial_prediction)

        print(prediction_probability)

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

