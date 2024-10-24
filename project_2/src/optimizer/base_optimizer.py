'''The base optimizer class that will include each learning strategy'''

class Optimizer:
    '''Base optimizer class'''
    def __init__(self, learning_rate=0.01, use_momentum=False, momentum=0.9):
        self.learning_rate = learning_rate
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.velocity = None
        self.use_mini_batch = False

    def step(self, params, gradients):
        '''Step method in optimization'''
        raise NotImplementedError('Subclasses is required for this method')