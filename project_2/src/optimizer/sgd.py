'''Stochastic gradient descent method'''
from .base_optimizer import Optimizer
import numpy as np

class SGD(Optimizer):
    '''Gradient descent class'''
    
    def __init__(self, learning_rate=0.001, use_momentum=False, momentum=0.9, use_mini_batch=True):
        '''
        Initialize the SGD optimizer
        
        Parameters:
        - learning_rate (float): The step size used in the update
        - use_momentum (bool): Flag to use momentum or not
        - momentum (float): Momentum factor
        - use_mini_batch (bool): Flag to indicate if mini-batch is used
        '''
        super().__init__(learning_rate, use_momentum, momentum)
        self.use_mini_batch = use_mini_batch
    
    def step(self, params, gradients):
        '''
        Perform one step of gradient descent optimization
        
        Parameters:
        - params (dict): Dictionary containing model parameters
        - gradients (dict): Dictionary containing the gradients of the model parameters
        '''
        if self.use_momentum:
            # Check if velocity has been initialized
            if self.velocity is None:
                self.velocity = {}
                # Initialize velocity for each parameter
                for key in params.keys():
                    self.velocity[key] = np.zeros_like(params[key])

            # Update parameters using momentum
            for key in params.keys():
                self.velocity[key] = self.momentum * self.velocity[key] + self.learning_rate * gradients[key]
                params[key] -= self.velocity[key]
        else:
            # Update parameters without using momentum
            for key in params.keys():
                params[key] -= self.learning_rate * gradients[key]
