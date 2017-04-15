# -*- coding: utf-8 -*-

from rorschach.utilities import Config, unpickle_data


class DimCalculator:

    # Entire shapes
    INPUT = 1
    LABELS = 2

    # Specific
    INPUT_WIDTH = 10
    INPUT_DEPTH = 11
    LABELS_WIDTH = 20
    LABELS_DEPTH = 21

    def __init__(self, predictor):
        self.predictor = predictor

    def get(self, value):
        check_set = self.find_set(value)

        if value == DimCalculator.INPUT or value == DimCalculator.LABELS:
            return check_set.shape

        if value in [DimCalculator.INPUT_WIDTH, DimCalculator.INPUT_DEPTH,
                     DimCalculator.LABELS_WIDTH, DimCalculator.LABELS_DEPTH]:
            return self.get_variable_dim(value, check_set)




        if value == DimCalculator.INPUT_WIDTH:
            return self.predictor.training_images_transformed.shape[-2]

        if value == DimCalculator.INPUT_DEPTH:
            return self.predictor.training_images_transformed.shape[-1]

        if value == DimCalculator.LABELS_WIDTH:
            return self.predictor.training_labels_transformed.shape[-2]

        if value == DimCalculator.LABELS_DEPTH:
            return self.predictor.training_labels_transformed.shape[-1]

        raise Exception('Unknown value')

    def find_set(self, value):
        if value == DimCalculator.INPUT_WIDTH or value == DimCalculator.INPUT_DEPTH:
            return self.predictor.training_images_transformed

        return self.predictor.training_labels_transformed

    def get_variable_dim(self, value, check_set):
        if len(check_set.shape) == 2:
            if value == DimCalculator.INPUT_WIDTH or value == DimCalculator.LABELS_WIDTH:
                return check_set.shape[-1]
            raise Exception('Asking for wrong axis')

        if len(check_set.shape) == 3:
            if value == DimCalculator.INPUT_WIDTH or value == DimCalculator.LABELS_WIDTH:
                return check_set.shape[-2]
            if value == DimCalculator.INPUT_DEPTH or value == DimCalculator.LABELS_DEPTH:
                return check_set.shape[-1]
            raise Exception('Asking for wrong axis')

        raise Exception('Unknown shape')
