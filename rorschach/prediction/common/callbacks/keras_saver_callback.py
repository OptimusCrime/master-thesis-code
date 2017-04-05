# -*- coding: utf-8 -*-

from rorschach.prediction.common.callbacks import BaseSaverCallback
from rorschach.utilities import Config


class KerasSaverCallback(BaseSaverCallback):

    def __init__(self):
        super().__init__()

    def run(self):
        if self.should_save_session():
            self.save_session(self.information['model'], self.data.get('epoch'))

    def save_session(self, model, epoch):
        # Add save to plot
        self.data.add_list('stores', epoch)

        # Save the weights via Keras helper method
        model.save_weights(Config.get_path('path.output', 'model.h5', fragment=Config.get('uid')))
