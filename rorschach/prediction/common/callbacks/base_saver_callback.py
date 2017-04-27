# -*- coding: utf-8 -*-

import copy

from rorschach.prediction.common.callbacks import BaseCallback
from rorschach.utilities import Config


class BaseSaverCallback(BaseCallback):

    def __init__(self):
        super().__init__()

    def run(self):
        pass

    def should_save_session(self):
        validation_loss = copy.copy(self.data.get('validate_loss'))

        # If we have no validations at all, skip the check
        if len(validation_loss) == 0:
            return False

        # 1. If this is the first epoch dump weights either way
        if len(validation_loss) == 1:
            return True

        # 2. If we have more than one epochs, check if the current loss is "equal" to the overall best loss we have
        # seen this far
        current_loss = validation_loss[-1]
        validation_loss.pop(-1)

        # We are really just comparing floats here
        return abs(min(validation_loss) - current_loss) <= 0.001
