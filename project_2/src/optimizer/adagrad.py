'''AdaGrad method'''
from .base_optimizer import Optimizer
import numpy as np

class AdaGrad(Optimizer):
    '''Gradient descent class with AdaGrad method'''
    def __init__(self,
                 learning_rate=0.001,
                 delta=1e-8,
                 use_momentum=False,
                 momentum=0.9,
                 use_mini_batch=False):
        # Initialize the base optimizer
        super().__init__(learning_rate, use_momentum, momentum)
        self.delta = delta
        self.accumulated_gradients = None
        self.use_mini_batch = use_mini_batch

    def step(self, params, gradients):
        '''
        Perform a step in the AdaGrad optimization method
        Momentum and mini batches are optional

        :param params: Dictionary containing model parameters.
        :param gradients: Dictionary containing gradients of the parameters.
        '''
        # Initialize accumulated gradients if not already done
        if self.accumulated_gradients is None:
            self.accumulated_gradients = {}
            for key in params.keys():
                self.accumulated_gradients[key] = np.zeros_like(params[key])

        # Check if momentum is used
        if self.use_momentum:
            if self.velocity is None:
                self.velocity = {}
                for key in params.keys():
                    self.velocity[key] = np.zeros_like(params[key])

            for key in params.keys():
                # Update accumulated gradients
                self.accumulated_gradients[key] += gradients[key] ** 2
                # Compute adaptive learning rate
                adaptive_lr = self.learning_rate / (np.sqrt(self.accumulated_gradients[key]) + self.delta)
                # Update velocity with momentum
                self.velocity[key] = self.momentum * self.velocity[key] + adaptive_lr * gradients[key]
                # Update parameters
                params[key] -= self.velocity[key]
        else:
            for key in params.keys():
                # Update accumulated gradients
                self.accumulated_gradients[key] += gradients[key] ** 2
                # Compute adaptive learning rate
                adaptive_lr = self.learning_rate / (np.sqrt(self.accumulated_gradients[key]) + self.delta)
                # Update parameters
                params[key] -= adaptive_lr * gradients[key]
