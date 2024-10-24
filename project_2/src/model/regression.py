'''Regression module'''
import numpy as np
import jax.numpy as jp

class Regression:
    '''Regression method'''
    def __init__(self, degree=1):
        self.degree = degree
        self.params = None

    def design_matrix(self,x):
        '''Making the design matrix'''
        return np.vander(x, N=self.degree+1, increasing=True)
        
    
