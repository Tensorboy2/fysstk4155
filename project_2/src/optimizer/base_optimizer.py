'''The base optimizer class that will include each learning strategy'''

class Optimizer:
    '''Base optimizer class'''

    def __init__(self, learning_rate=0.01, use_momentum=False, momentum=0.9):
        '''
        Initialize the optimizer with the given parameters.

        :param learning_rate: Learning rate for the optimization.
        :param use_momentum: Boolean to decide if momentum is to be used.
        :param momentum: Momentum factor to be used if use_momentum is True.
        '''
        self.learning_rate = learning_rate
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.velocity = None
        self.use_mini_batch = False

    def step(self, params, gradients):
        '''
        Perform a single optimization step.

        :param params: Model parameters to be updated.
        :param gradients: Gradients used for updating the parameters.
        '''
        raise NotImplementedError('Subclasses must implement this method')
