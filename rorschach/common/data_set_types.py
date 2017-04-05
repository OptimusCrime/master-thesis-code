# -*- coding: utf-8 -*-


class DataSetTypes:

    LETTER_SET = 0
    TRAINING_SET = 1
    VALIDATE_SET = 2
    TEST_SET = 3

    LABELS = 'labels'
    IMAGES = 'images'

    def __init__(self):
        pass

    @staticmethod
    def type_to_keyword(data_type):
        if data_type == DataSetTypes.LETTER_SET:
            return 'letter'

        if data_type == DataSetTypes.TEST_SET:
            return 'test'

        if data_type == DataSetTypes.TRAINING_SET:
            return 'training'

        return 'validate'
