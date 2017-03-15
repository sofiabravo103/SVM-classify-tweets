import sys
import random


class IAModel():

    def __init__(self, ts_x, ts_y):
        self.model = None
        self.ts_x = ts_x
        self.ts_y = ts_y

    def balance(self, print_report=True):
        pass

    def train(self):
        if not self.model:
            raise Exception('No model defined.')

        print('Training model ......... ', end='')

        self.model.fit(self.ts_x, self.ts_y)

        print('\rModel trained successfully.\n')

    def eval(self, rep):
        if not self.model:
            raise Exception('No model defined.')
        else:
            return self.model.predict(rep)
