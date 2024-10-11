'''ffnn module'''
import numpy as np
import jax.numpy as jnp
from jax import grad, jit
np.random.seed(42)

class FFNN:
    '''
    Feed Forward Neural Network.
    '''
    def __init__(self,
                 hidden_size,
                 num_hidden_layers,
                 input_size=None,
                 output_size=None,
                 learning_rate=0.1,
                 activation_function=None,
                 output_function=None):

        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_hidden_layers = num_hidden_layers
        self.activation_function = self.sigmoid if activation_function is None else activation_function
        self.output_function = self.straight_output if output_function is None else output_function

        self.loss = None
        self.optimizer = None

        self.input_layer = np.random.normal(0,1,(self.input_size,hidden_size))
        self.hidden_layers = np.random.normal(0, 1,(num_hidden_layers,hidden_size, hidden_size))
        self.output_layer = np.random.normal(0,1,(hidden_size,self.output_size))
        self.params = [self.input_layer,
                       self.hidden_layers,
                       self.output_layer]
        self.paramsi = self.params

    def sigmoid(self,x):
        ''' 
        Sigmoid activation function.
        '''
        return 1/(1+jnp.exp(-x))

    def relu(self,x):
        '''
        Relu activation function.
        Try not to use this function without 
        scaling down initial wights as they might explode.
        '''
        return jnp.maximum(0,x)

    def leaky_relu(self,x,alpha=0.01):
        '''
        Leaky relu activation function.
        Try not to use this function without 
        scaling down initial wights as they might explode.
        '''
        return jnp.where(x>0,x,alpha*x)

    def set_activation_function(self,activation_function):
        '''
        Set activation function
        '''
        self.activation_function = activation_function

    def straight_output(self,x):
        '''
        No output activation function for the output layer
        '''
        return x

    def set_optimizer(self,optimizer):
        '''
        Set optimizer
        '''
        self.optimizer = optimizer

    def forward(self,x):
        '''
        Feed forward method.
        '''
        input_layer, hidden_layers, output_layer = self.params
        # First Layer
        x = self.activation_function(jnp.dot(x.T,input_layer))

        # Optional multiple hidden layers
        if (self.num_hidden_layers!=0):
            for hidden in hidden_layers:
                x = self.activation_function(jnp.dot(x,hidden))

        # Last layer
        x = self.output_function(jnp.dot(x,output_layer))

        return x

    def cost(self,params,y,x):
        '''
        Mean square error
        '''
        input_layer, hidden_layers, output_layers = params
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layers

        # Compute forward pass
        predictions = self.forward(x)

        # Compute Mean Squared Error
        loss = jnp.mean((y - predictions) ** 2)
        self.loss = loss
        return loss

    def backprop(self,y,x):
        '''
        Step in gradient method
        '''
        params=self.params

        if self.optimizer=='GD':
            gradient = grad(self.cost)
            gradients = gradient(params,y,x)
            self.cost(params,y,x)
            self.params = [p - self.learning_rate * g for p, g in zip(params, gradients)]
        
        elif self.optimizer=='SGD':
            M = 5
            m = int(self.input_size/M)
            # print(m)
            for _ in range(m):
                random_index = M*np.random.randint(0,m)
                xi = x[random_index:random_index+M]
                yi = y[random_index:random_index+M]
                # print(xi.shape)
                # print(params[0][random_index:random_index+M,:].shape)
                # print(jnp.dot(xi.T,params[0][random_index:random_index+M,:]).shape)
                self.paramsi = [params[0][random_index:random_index+M,:],
                           params[1],
                           params[2][:,random_index:random_index+M]]
                gradient = grad(self.cost)
                gradients = gradient(self.params,yi,xi)
                self.cost(self.paramsi,yi,xi)
                self.params = [p - self.learning_rate * g for p, g in zip(params, gradients)]


if __name__ == '__main__':

    input_size_ = np.random.randint(10,100)
    output_size_ = np.random.randint(10,100)
    hidden_size_ = np.random.randint(10,100)
    num_hidden_layers_ = np.random.randint(10,100)

    model = FFNN(hidden_size=hidden_size_,
                 num_hidden_layers=num_hidden_layers_,
                 input_size=input_size_,
                 output_size=output_size_,
                 learning_rate=0.1)
    
    model.set_optimizer('GD')

    x_ = np.random.rand(input_size_)
    y_ = np.random.rand(output_size_)

    for i in range(100):
        model.backprop(y=y_,x=x_)
        print(f'model loss: {model.loss:.6f} for epoch: {i}')
        if (model.loss<1e-6):
            break
