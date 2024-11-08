'''Gradient descent method'''
from .base_optimizer import Optimizer
import numpy as np

class GD(Optimizer):
    '''Gradient descent class'''
    def __init__(self, learning_rate=0.001, use_momentum=False, momentum=0.9):
        super().__init__(learning_rate, use_momentum, momentum)

    def step(self, params, gradients):
        '''Perform a step in the gradient descent optimization'''
        
        # Check if momentum is used
        if self.use_momentum:
            # Initialize velocity term if it's not already initialized
            if self.velocity is None:
                self.velocity = {}
                for key in params.keys():
                    self.velocity[key] = np.zeros_like(params[key])

            # Update parameters with momentum
            for key in params.keys():
                self.velocity[key] = self.momentum * self.velocity[key] + self.learning_rate * gradients[key]
                params[key] -= self.velocity[key]
                # print(gradients[key].shape)
        
        else:
            # Update parameters without momentum
            for key in params.keys():
                params[key] -= self.learning_rate * gradients[key]
