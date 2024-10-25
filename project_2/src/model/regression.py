'''Regression module'''
import numpy as np
import jax.numpy as jp

class Regression:
    '''Regression method'''
    def __init__(self, x, degree=1):
        self.degree = degree
        self.params = None
        self.X = self.design_matrix(x)
        self.beta = np.random.randn(self.X.shape[0])

    def design_matrix(self,x):
        '''Making the design matrix'''
        return np.vander(x, N=self.degree+1, increasing=True)

    def predict(self,X,beta):
        '''Making prediction form current beta,
        similar to NN forward method in the gradient process.'''
        return jp.dot(beta,X.T)

    def mse(self,y,X,beta):
        '''mse for prediction'''
        pred = self.predict(X,beta)
        return jp.mean((y - pred)**2)
