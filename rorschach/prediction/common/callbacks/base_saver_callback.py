# -*- coding: utf-8 -*-

from rorschach.prediction.common.callbacks import BaseCallback


class BaseSaverCallback(BaseCallback):

    def __init__(self):
        super().__init__()

    def run(self):
        pass

    def should_save_session(self):
        # If we have no validations at all, skip the check
        if len(self.data.get('validate_loss')) == 0:
            return False

        # 1. If this is the first epoch dump weights either way
        if len(self.data.get('validate_loss')) == 1:
            return True

        # We are really just comparing floats here
        return abs(min(self.data.get('validate_loss')) - self.data.get('validate_loss')[-1]) <= 0.001
