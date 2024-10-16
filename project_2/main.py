'''
The main file of the project.
This file should run all nessesary code to produce all results.
'''
import numpy as np
from data.prepare_data import prepare_data

from src.FFNN import FFNN
from src.torch_FFNN import torch_ffnn, train

optimizers = ['GD', 'GD with momentum',
               'SGD', 'SGD with momentum',
                 'AdaGrad with GD', 'AdaGrad with GD with momentum',
                 'AdaGrad with SGD', 'AdaGrad with SGD with momentum',
                 'RMSprop', 'Adam']

learning_rates = [0.1,0.01,0.001]

regressions = ['OLS', 'Ridge']

tasks = ['regression', 'classify']

def f(x_,coeff):
    deg = len(coeff)
    poly = 0
    for i in range(deg):
        poly += coeff[i]*x_**(i)
    return poly

def part_1():
    '''Regression (fitting a continous function).

    In this function we will compare the regression methods form normal regression and methods from a Neural Network.
    We will use backpropagation with different learning strategies, trying out different learing rates and estimate their preformence by the MSE and R2.
    '''
    n_points = 100
    deg = 3
    coefficients = np.random.rand(deg)
    x = np.linspace(-1,1,n_points)
    y = f(x,coeff=coefficients)




def part_2():
    '''Classisfication.

    In this function we will compare the classification of a logisitc regression method and the use of a Feed Forward Neural Network.

    '''
    x_train, x_test, y_train, y_test = prepare_data()

    input_size = ?
    output_size = ?
    hidden_size = ?
    num_hidden_layers = ?
    own_ff = FFNN()
    torch_ff = torch_ffnn()
    own_ff.train()
    train(torch_ff,criterion=,optimizer=,train_loader=,epochs=)

if __name__ == '__main__':
    part_1()
    part_2()
