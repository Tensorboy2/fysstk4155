'''Activation functions'''
import jax.numpy as jp

class LossFunctions:
    '''Loss functions for optimization'''

    @staticmethod
    def mse(prediction, targets):
        '''Mean square error'''
        return jp.mean((prediction - targets) ** 2)

    @staticmethod
    def cross_entropy(predictions, targets):
        '''Cross entropy cost function'''
        return -jp.sum(targets * jp.log(predictions + 1e-9)) / targets.shape[0]
