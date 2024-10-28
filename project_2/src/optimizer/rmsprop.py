'''RMSprop method'''
from .base_optimizer import Optimizer
import numpy as np

class RMSprop(Optimizer):
    '''RMSprop class'''
    def __init__(self,
                 learning_rate=0.01,
                 delta = 1e-8,
                 use_momentum=False,
                 rho=0.99,
                 use_mini_batch=True):
        super().__init__(learning_rate, use_momentum)
        self.delta = delta
        self.rho = rho
        self.squared_gradients = None
        self.use_mini_batch=use_mini_batch

    def step(self, params, gradients):
        '''step in RMSprop method optimization'''
        if self.squared_gradients is None:
            self.squared_gradients = {}
            for key in params.keys():
                self.squared_gradients[key] = np.zeros_like(params[key])
        
        
        for key in params.keys():
            self.squared_gradients[key] = self.rho*self.squared_gradients[key] + (1-self.rho)*gradients[key] ** 2
            adaptive_lr = self.learning_rate / (np.sqrt(self.squared_gradients[key]) + self.delta)
            params[key] -= adaptive_lr * gradients[key]
