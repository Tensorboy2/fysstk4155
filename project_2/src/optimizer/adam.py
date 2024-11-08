'''Adam method'''
from .base_optimizer import Optimizer
import numpy as np

class Adam(Optimizer):
    '''Adam class'''
    
    def __init__(self, learning_rate=0.01, delta=1e-8, use_momentum=False, use_mini_batch=True):
        super().__init__(learning_rate, use_momentum)
        self.delta = delta
        self.beta1 = 0.99
        self.beta2 = 0.999
        self.first_momentum = None
        self.second_momentum = None
        self.iter = 0
        self.use_mini_batch = use_mini_batch

    def step(self, params, gradients):
        '''
        Step in Adam method optimization
        

        :param params: Dictionary containing model parameters.
        :param gradients: Dictionary containing gradients of the parameters.
        '''
        
        # Initialize first and second momentums if not already initialized
        if self.first_momentum is None:
            self.first_momentum = {}
            self.second_momentum = {}
            for key in params.keys():
                self.first_momentum[key] = np.zeros_like(params[key])
                self.second_momentum[key] = np.zeros_like(params[key])
        
        self.iter += 1
        for key in params.keys():
            # Update first and second momentums
            self.first_momentum[key] = self.beta1 * self.first_momentum[key] + (1 - self.beta1) * gradients[key]
            self.second_momentum[key] = self.beta2 * self.second_momentum[key] + (1 - self.beta2) * gradients[key] ** 2

            # Compute bias-corrected first and second momentums
            first_term = self.first_momentum[key] / (1.0 - self.beta1 ** self.iter)
            second_term = self.second_momentum[key] / (1.0 - self.beta2 ** self.iter)

            # Update parameters
            params[key] -= self.learning_rate * first_term / (np.sqrt(second_term) + self.delta)
