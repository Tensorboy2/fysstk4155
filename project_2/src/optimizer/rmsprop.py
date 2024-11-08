'''RMSprop method'''
from .base_optimizer import Optimizer
import numpy as np

class RMSprop(Optimizer):
    '''RMSprop class for implementing the RMSprop optimization algorithm'''

    def __init__(self, learning_rate=0.01, delta=1e-8, use_momentum=False, rho=0.99, use_mini_batch=True):
        '''
        Initialize the RMSprop optimizer.
        
        :param learning_rate: Learning rate for the optimizer.
        :param delta: Small value to avoid division by zero.
        :param use_momentum: Flag to use momentum in the optimizer.
        :param rho: Decay rate for the moving average of squared gradients.
        :param use_mini_batch: Flag to indicate use of mini-batch updates.
        '''
        super().__init__(learning_rate, use_momentum)
        self.delta = delta
        self.rho = rho
        self.squared_gradients = None
        self.use_mini_batch = use_mini_batch

    def step(self, params, gradients):
        '''
        Perform a single optimization step.
        
        :param params: Dictionary containing model parameters.
        :param gradients: Dictionary containing gradients of the parameters.
        '''
        # Initialize squared gradients if they are None
        if self.squared_gradients is None:
            self.squared_gradients = {}
            for key in params.keys():
                self.squared_gradients[key] = np.zeros_like(params[key])
        
        # Update parameters
        for key in params.keys():
            self.squared_gradients[key] = self.rho * self.squared_gradients[key] + (1 - self.rho) * gradients[key] ** 2
            adaptive_lr = self.learning_rate / (np.sqrt(self.squared_gradients[key]) + self.delta)
            params[key] -= adaptive_lr * gradients[key]
