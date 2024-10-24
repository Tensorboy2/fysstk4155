'''AdaGrad method'''
from .base_optimizer import Optimizer
import numpy as np

class AdaGrad(Optimizer):
    '''Gradient decent class'''
    def __init__(self,
                 learning_rate=0.001,
                 delta = 1e-8,
                 use_momentum=False,
                 momentum=0.9,
                 use_mini_batch=False):
        super().__init__(learning_rate, use_momentum, momentum)
        self.delta = delta
        self.accumulated_gradients = None
        self.use_mini_batch=use_mini_batch

    def step(self, params, gradients):
        '''step in AdaGrad method optimization
        Momentum and mini batches are optional
        '''
        if self.accumulated_gradients is None:
            self.accumulated_gradients = {}
            for key in params.keys():
                self.accumulated_gradients[key] = np.zeros_like(params[key])
        
        if self.use_momentum:
            if self.velocity is None:
                self.velocity = {}
                for key in params.keys():
                    self.velocity[key] = np.zeros_like(params[key])

            for key in params.keys():
                self.accumulated_gradients[key] += gradients[key] ** 2
                adaptive_lr = self.learning_rate / (np.sqrt(self.accumulated_gradients[key]) + self.delta)
                self.velocity[key] = self.momentum*self.velocity[key] + adaptive_lr*gradients[key]
                params[key] -= self.velocity[key]
        else:
            for key in params.keys():
                self.accumulated_gradients[key] += gradients[key] ** 2
                adaptive_lr = self.learning_rate / (np.sqrt(self.accumulated_gradients[key]) + self.delta)
                params[key] -= adaptive_lr * gradients[key]
