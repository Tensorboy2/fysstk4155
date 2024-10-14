'''ffnn module'''
import numpy as np
import jax.numpy as jnp
from jax import grad, jit
# import matplotlib.pyplot as plt
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
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_hidden_layers = num_hidden_layers
        self.activation_function = self.sigmoid if activation_function is None else activation_function
        self.output_function = self.straight_output if output_function is None else output_function

        self.loss = None
        self.optimizer = None

        # self.input_layer = np.random.rand(self.input_size,hidden_size)
        # self.hidden_layers = np.random.rand(num_hidden_layers,hidden_size, hidden_size)
        # self.output_layer = np.random.rand(hidden_size,self.output_size)
        self.input_layer = np.random.normal(0,1,(self.input_size,hidden_size))
        self.hidden_layers = np.random.normal(0, 1,(num_hidden_layers,hidden_size, hidden_size))
        self.output_layer = np.random.normal(0,1,(hidden_size,self.output_size))

        self.params = [self.input_layer,
                       self.hidden_layers,
                       self.output_layer]
        
        # self.paramsi = self.params

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

    def forward(self,x,params):
        '''
        Feed forward method.
        '''
        input_layer, hidden_layers, output_layer = params
        # First Layer
        # print(x.shape)
        # print(input_layer.shape)
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
        predictions = self.forward(x,params)
        loss = jnp.mean((y - predictions) ** 2)
        self.loss = loss
        return loss
    
    def backprop(self,y,x):
        '''
        Step in gradient method
        '''
        params=self.params
        match self.optimizer:
            case 'GD':
                gradients = grad(self.cost,argnums=0)(params,y,x)
                # gradients = gradient(params,y,x)
                self.params = [p - self.learning_rate * g for p, g in zip(params, gradients)]
                self.cost(params,y,x)

            case 'GD with momentum':
                if not hasattr(self,'velocity'):
                    self.velocity = [np.zeros_like(self.input_layer),
                                     np.zeros_like(self.hidden_layers),
                                     np.zeros_like(self.output_layer)]
                self.gamma = 0.9

                gradients = grad(self.cost,argnums=0)(params,y,x)
                # gradients = gradient(params,y,x)
                self.params = [p - self.learning_rate * g for p, g in zip(params, gradients)]

                for j in range(3):
                    self.velocity[j] = (self.gamma*self.velocity[j]
                                        + self.learning_rate *gradients[j])
                    self.params[j] -= self.velocity[j]
                self.cost(params,y,x)

            case 'SGD with momentum':
                if not hasattr(self,'velocity'):
                    self.velocity = [np.zeros_like(self.input_layer),
                                     np.zeros_like(self.hidden_layers),
                                     np.zeros_like(self.output_layer)]
                self.gamma = 0.9
                M = 5
                m = int(self.input_size/M)
                for _ in range(m):
                    random_index = M*np.random.randint(0,m)
                    xi = x[random_index:random_index+M]
                    yi = y[random_index:random_index+M]

                    paramsi_0 = params[0][random_index:random_index+M,:]
                    paramsi_2 = params[2][:,random_index:random_index+M]

                    paramsi = [paramsi_0, params[1], paramsi_2]

                    gradients = grad(self.cost,argnums=0)(paramsi,yi,xi)
                    # gradients = gradient(paramsi,yi,xi)

                    # Update input layer:
                    self.velocity[0][random_index:random_index+M,:] = self.gamma * self.velocity[0][random_index:random_index+M,:] + self.learning_rate*gradients[0]

                    self.params[0] -= self.velocity[0]

                    # Update hidden layer:
                    self.velocity[1]= self.gamma * self.velocity[1] + self.learning_rate*gradients[1]

                    self.params[1] -= self.velocity[1]

                    # Update output layer:
                    self.velocity[2][:,random_index:random_index+M] = self.gamma * self.velocity[2][:,random_index:random_index+M] + self.learning_rate*gradients[2]

                    self.params[2] -= self.velocity[2]

                self.cost(params,y,x)

            case 'SGD':
                M = 5
                m = int(self.input_size/M)
                for _ in range(m):
                    random_index = M*np.random.randint(0,m)
                    xi = x[random_index:random_index+M]
                    yi = y[random_index:random_index+M]

                    paramsi_0 = params[0][random_index:random_index+M,:]
                    paramsi_2 = params[2][:,random_index:random_index+M]

                    paramsi = [paramsi_0, params[1], paramsi_2]

                    gradients = grad(self.cost,argnums=0)(paramsi,yi,xi)
                    # gradients = gradient(paramsi,yi,xi)

                    self.params[0][random_index:random_index+M,:] -= self.learning_rate*gradients[0]
                    self.params[1] -= self.learning_rate * gradients[1]
                    self.params[2][:,random_index:random_index+M] -= self.learning_rate*gradients[2]
                self.cost(params,y,x)

            case 'AdaGrad with GD':
                if not hasattr(self,'sum_squared_gradients'):
                    self.sum_squared_gradients = [np.zeros_like(self.input_layer),
                                     np.zeros_like(self.hidden_layers),
                                     np.zeros_like(self.output_layer)]
                delta = 1e-8
                gradients = grad(self.cost,argnums=0)(params,y,x)
                for j in range(3):
                    self.sum_squared_gradients[j] += gradients[j]**2
                    self.params[j] -= self.learning_rate*gradients[j]/(np.sqrt(self.sum_squared_gradients[j])+delta)
                self.cost(params,y,x)

            case 'AdaGrad with GD with momentum':
                if not hasattr(self,'sum_squared_gradients'):
                    self.sum_squared_gradients = [np.zeros_like(self.input_layer),
                                     np.zeros_like(self.hidden_layers),
                                     np.zeros_like(self.output_layer)]
                    self.velocity = [np.zeros_like(self.input_layer),
                                     np.zeros_like(self.hidden_layers),
                                     np.zeros_like(self.output_layer)]
                delta = 1e-8
                gamma = 0.9
                learning_rate = self.learning_rate
                gradients = grad(self.cost,argnums=0)(params,y,x)
                for j in range(3):
                    self.sum_squared_gradients[j] += gradients[j]**2
                    self.velocity[j] = gamma*self.velocity[j] + learning_rate*gradients[j]/(np.sqrt(self.sum_squared_gradients[j])+delta)
                    self.params[j] -= self.velocity[j]
                self.cost(params,y,x)

            case 'AdaGrad with SGD':
                pass

            case 'Adagrad with SGD with momentum':
                pass

            case 'RMSprop':
                pass
            case 'Adam':
                if not hasattr(self,'iter'):
                    self.iter = 0

                first_moment = [0,0,0]
                second_moment = [0,0,0]
                self.iter +=1
                beta1 = 0.9
                beta2 = 0.99
                delta = 1e-8
                M = 5
                m = int(self.input_size/M)
                for _ in range(m):
                    random_index = M*np.random.randint(0,m)
                    xi = x[random_index:random_index+M]
                    yi = y[random_index:random_index+M]

                    paramsi_0 = params[0][random_index:random_index+M,:]
                    paramsi_2 = params[2][:,random_index:random_index+M]

                    paramsi = [paramsi_0, params[1], paramsi_2]

                    gradients = grad(self.cost,argnums=0)(paramsi,yi,xi)
                    first_term = np.zeros((3,M,self.hidden_size))
                    second_term = np.zeros((3,M,self.hidden_size))
                    for j in range(3):
                        first_moment[j] = beta1*first_moment[j] + (1-beta1)*gradients[j]
                        second_moment[j] = beta2*second_moment[j] + (1-beta2)*gradients[j]*gradients[j]
                        first_term[j] = first_moment[j]/(1.0-beta1**self.iter)
                        second_term[j] = second_moment[j]/(1.0-beta2**self.iter)

                    self.params[0][random_index:random_index+M,:] -= self.learning_rate*first_term[0]/(np.sqrt(second_term[0])+delta)
                    self.params[1] -= self.learning_rate*first_term[1]/(np.sqrt(second_term[1])+delta)
                    self.params[2][:,random_index:random_index+M] -= self.learning_rate*first_term[2]/(np.sqrt(second_term[2])+delta)
                
                self.cost(params,y,x)
                

    def train(self,x,y,epochs=10,threshold=1e-6):
        '''
        Training method.
        '''
        for epoch in range(epochs):
            self.backprop(y, x)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {self.loss:.6f}')
            if self.loss < threshold:
                print('Convergence reached.')
                break
            elif np.isnan(self.loss):
                print('Loss turned nan')
                break

if __name__ == '__main__':
    min = 10
    max = 100
    input_size_ = np.random.randint(min, max)
    output_size_ = np.random.randint(min, max)
    hidden_size_ = np.random.randint(min, max)
    num_hidden_layers_ = np.random.randint(min, max)

    model = FFNN(hidden_size=hidden_size_,
                num_hidden_layers=num_hidden_layers_,
                input_size=input_size_,
                output_size=output_size_,
                learning_rate=0.1)

    model.set_optimizer('Adam')
    model.set_activation_function(model.sigmoid)
    model.output_function(model.straight_output)

    x_ = np.random.rand(input_size_)
    y_ = np.random.rand(output_size_)
    model.train(x=x_, y=y_, epochs=50,threshold=1e-2)
