'''Stochastic gradient decent method'''
from .base_optimizer import Optimizer
import numpy as np

class SGD(Optimizer):
    '''Gradient decent class'''
    def __init__(self, learning_rate=0.001, use_momentum=False, momentum=0.9):
        super().__init__(learning_rate, use_momentum, momentum)
        self.use_mini_batch=True
    def step(self, params, gradients):
        '''step in gradient decent method optimization'''
        if self.use_momentum:
            if self.velocity is None:
                self.velocity = {}
                for key in params.keys():
                    self.velocity[key] = np.zeros_like(params[key])

            for key in params.keys():
                self.velocity[key] = self.momentum*self.velocity[key] + self.learning_rate*gradients[key]
                params[key] -= self.velocity[key]
        else:
            for key in params.keys():
                params[key] -= self.learning_rate * gradients[key]
