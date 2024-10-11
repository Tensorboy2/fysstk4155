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
                 learning_rate=0.1):

        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.loss = None
        self.optimizer = None

        self.num_hidden_layers = num_hidden_layers


        self.input_layer = np.random.normal(0,1,(self.input_size,hidden_size))
        self.hidden_layers = np.random.normal(0,1,(num_hidden_layers,hidden_size,hidden_size))
        self.output_layer = np.random.normal(0,1,(hidden_size,self.output_size))
        self.params = [self.input_layer,
                       self.hidden_layers,
                       self.output_layer]

    def sigmoid(self,x):
        ''' 
        Sigmoid activation function.
        '''
        return 1/(1+jnp.exp(-x))

    def set_optimizer(self,optimizer):
        '''
        Set optimizer
        '''
        self.optimizer = optimizer

    def forward(self,x):
        '''
        Feed forward method.
        '''

        # First Layer
        x = self.sigmoid(jnp.dot(x.T,self.input_layer))

        # Optional multiple hidden layers
        if (self.num_hidden_layers!=0):
            for hidden in self.hidden_layers:
                x = self.sigmoid(jnp.dot(x,hidden))

        # Last layer
        x = self.sigmoid(jnp.dot(x,self.output_layer))

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

    def step(self,y,x):
        '''
        Step in gradient method
        '''
        params=self.params
        if self.optimizer=='GD':
            gradient = grad(self.cost)
            gradients = gradient(params,y,x)
            self.cost(params,y,x)

            self.params = [p - self.learning_rate * g for p, g in zip(params, gradients)]


if __name__ == '__main__':

    input_size_ = np.random.randint(10,100)
    output_size_ = np.random.randint(10,100)
    hidden_size_ = np.random.randint(10,100)
    num_hidden_layers_ = np.random.randint(10,100)


    model = FFNN(hidden_size=hidden_size_,
                 num_hidden_layers=num_hidden_layers_,
                 input_size=input_size_,
                 output_size=output_size_)
    model.set_optimizer('GD')
    x_ = np.random.rand(input_size_)
    y_ = np.random.rand(output_size_)
    for i in range(100):
        model.step(y=y_,x=x_)
        print(model.loss)
