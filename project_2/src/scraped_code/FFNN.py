'''ffnn module'''
import numpy as np
import jax.numpy as jnp
from jax import grad
import sys
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
        self.input_weights = np.random.normal(0,1,(self.input_size,hidden_size))
        self.input_bias = np.random.normal(0,1,hidden_size)
        self.input_layer = [self.input_weights, self.input_bias]

        self.hidden_weights = np.random.normal(0, 1,(num_hidden_layers,hidden_size, hidden_size))
        self.hidden_bias = np.random.normal(0, 1,(num_hidden_layers, hidden_size))
        self.hidden_layers = [self.hidden_weights, self.hidden_bias]


        self.output_weights = np.random.normal(0,1,(hidden_size,self.output_size))
        self.output_bias = np.random.normal(0,1,self.output_size)
        self.output_layer = [self.output_weights, self.output_bias]

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
    def classify_output(self,x):
        '''
        output activation function gives only 0 or 1
        '''
        x = self.sigmoid(x)
        if x>0.5:
            return 1
        else:
            return 0

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
        x = self.activation_function(jnp.dot(x,input_layer[0])+input_layer[1])
        # print(len(hidden_layers[0]))
        if (self.num_hidden_layers!=0):
            for k in range(self.num_hidden_layers):
                x = self.activation_function(jnp.dot(x,hidden_layers[0][k])+hidden_layers[1][k])
        # print(output_layer[0].shape)
        # print(output_layer[1].shape)
        # print(x.shape)
        # print(jnp.dot(x,output_layer[0]))
        x = self.output_function(jnp.dot(x,output_layer[0])+output_layer[1])
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
                for j in range(3):
                    self.params[j][0] -= self.learning_rate * gradients[j][0] # Weight update
                    self.params[j][1] -= self.learning_rate * gradients[j][1] # Bias update

                self.cost(params,y,x)

            case 'GD with momentum':
                if not hasattr(self,'velocity'):
                    self.weight_velocity = [np.zeros_like(params[0][0]),
                                     np.zeros_like(params[1][0]),
                                     np.zeros_like(params[2][0])]
                    self.bias_velocity = [np.zeros_like(params[0][1]),
                                     np.zeros_like(params[1][1]),
                                     np.zeros_like(params[2][1])]
                self.gamma = 0.9

                gradients = grad(self.cost,argnums=0)(params,y,x)

                for j in range(3):
                    self.weight_velocity[j] = (self.gamma*self.weight_velocity[j] + self.learning_rate *gradients[j][0])
                    self.params[j][0] -= self.weight_velocity[j]
                    self.bias_velocity[j] = (self.gamma*self.bias_velocity[j] + self.learning_rate *gradients[j][1])
                    self.params[j][1] -= self.bias_velocity[j]
                self.cost(params,y,x)

            case 'SGD with momentum':
                if not hasattr(self,'velocity'):
                    self.weight_velocity = [np.zeros_like(params[0][0]),
                                     np.zeros_like(params[1][0]),
                                     np.zeros_like(params[2][0])]
                    self.bias_velocity = [np.zeros_like(params[0][1]),
                                     np.zeros_like(params[1][1]),
                                     np.zeros_like(params[2][1])]
                self.gamma = 0.9
                
                M = min(5,self.input_size)
                m = int(self.input_size/M)
                for _ in range(m):
                    random_index = M*np.random.randint(0,m)
                    xi = x[random_index:random_index+M]
                    yi = y[random_index:random_index+M]

                    paramsi = [[params[0][0][random_index:random_index+M,:],params[0][1]],
                                params[1],
                                [params[2][0][:,random_index:random_index+M],params[2][1][random_index:random_index+M]]]

                    gradients = grad(self.cost,argnums=0)(paramsi,yi,xi)
                    # gradients = gradient(paramsi,yi,xi)

                    # Update input layer:
                    self.weight_velocity[0][random_index:random_index+M,:] = self.gamma * self.weight_velocity[0][random_index:random_index+M,:] + self.learning_rate*gradients[0][0]
                    self.params[0][0] -= self.weight_velocity[0]

                    self.bias_velocity[0] = self.gamma * self.bias_velocity[0] + self.learning_rate*gradients[0][0]
                    self.params[0][1] -= self.bias_velocity[0]

                    # Update hidden layer:
                    self.weight_velocity[1]= self.gamma * self.weight_velocity[1] + self.learning_rate*gradients[1][0]
                    self.params[1][0] -= self.weight_velocity[1]
                    self.bias_velocity[1]= self.gamma * self.bias_velocity[1] + self.learning_rate*gradients[1][1]
                    self.params[1][1] -= self.bias_velocity[1]

                    # Update output layer:
                    self.weight_velocity[2][:,random_index:random_index+M] = self.gamma * self.weight_velocity[2][:,random_index:random_index+M] + self.learning_rate*gradients[2][0]
                    self.params[2][0] -= self.weight_velocity[2]

                    self.bias_velocity[2][random_index:random_index+M] = self.gamma * self.bias_velocity[2][random_index:random_index+M] + self.learning_rate*gradients[2][1]
                    self.params[2][1] -= self.bias_velocity[2]

                self.cost(params,y,x)

            case 'SGD':
                M = min(5,self.input_size)
                m = int(self.input_size/M)
                for _ in range(m):
                    random_index = M*np.random.randint(0,m)
                    xi = x[random_index:random_index+M]
                    yi = y[random_index:random_index+M]

                    paramsi = [[params[0][0][random_index:random_index+M,:],params[0][1]],
                                params[1],
                                [params[2][0][:,random_index:random_index+M],params[2][1][random_index:random_index+M]]]

                    gradients = grad(self.cost,argnums=0)(paramsi,yi,xi)
                    # gradients = gradient(paramsi,yi,xi)

                    # Update input layer:
                    self.params[0][0][random_index:random_index+M,:] -= self.learning_rate*gradients[0][0]
                    self.params[0][1] -= self.learning_rate*gradients[0][0]

                    # Update hidden layer:
                    self.params[1][0] -= self.learning_rate*gradients[1][0]
                    self.params[1][1] -= self.learning_rate*gradients[1][1]

                    # Update output layer:
                    self.params[2][0][:,random_index:random_index+M] -= self.learning_rate*gradients[2][0]
                    self.params[2][1][random_index:random_index+M] -= self.learning_rate*gradients[2][1]

                self.cost(params,y,x)

            case 'AdaGrad with GD':
                if not hasattr(self,'sum_squared_gradients'):
                    self.sum_squared_gradients = [[np.zeros_like(self.input_layer[0]),np.zeros_like(self.input_layer[0])],
                                     [np.zeros_like(self.hidden_layers[0]),np.zeros_like(self.hidden_layers[1])],
                                     [np.zeros_like(self.output_layer[0]),np.zeros_like(self.output_layer[1])]]
                delta = 1e-8
                gradients = grad(self.cost,argnums=0)(params,y,x)

                for j in range(3):

                    self.sum_squared_gradients[j][0] += gradients[j][0]**2
                    self.params[j][0] -= self.learning_rate * gradients[j][0]/(np.sqrt(self.sum_squared_gradients[j][0])+delta) # Weight update

                    self.sum_squared_gradients[j][1] += gradients[j][1]**2
                    self.params[j][1] -= self.learning_rate * gradients[j][1]/(np.sqrt(self.sum_squared_gradients[j][1])+delta) # Bias update
                
                self.cost(params,y,x)

            case 'AdaGrad with GD with momentum':
                if not hasattr(self,'sum_squared_gradients'):
                    self.sum_squared_gradients = [[np.zeros_like(self.input_layer[0]),np.zeros_like(self.input_layer[0])],
                                     [np.zeros_like(self.hidden_layers[0]),np.zeros_like(self.hidden_layers[1])],
                                     [np.zeros_like(self.output_layer[0]),np.zeros_like(self.output_layer[1])]]
                    self.velocity =  [[np.zeros_like(self.input_layer[0]),np.zeros_like(self.input_layer[0])],
                                     [np.zeros_like(self.hidden_layers[0]),np.zeros_like(self.hidden_layers[1])],
                                     [np.zeros_like(self.output_layer[0]),np.zeros_like(self.output_layer[1])]]
                delta = 1e-8
                gamma = 0.9
                learning_rate = self.learning_rate
                gradients = grad(self.cost,argnums=0)(params,y,x)
                for j in range(3):
                    for jj in range(2):
                        self.sum_squared_gradients[j][jj] += gradients[j][jj]**2
                        self.velocity[j][jj] = gamma*self.velocity[j][jj] + learning_rate*gradients[j][jj]/(np.sqrt(self.sum_squared_gradients[j][jj])+delta)
                        self.params[j][jj] -= self.velocity[j][jj]
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
                if not hasattr(self, 'first_moment'):
                    self.first_moment = [[np.zeros_like(self.input_layer[0]),np.zeros_like(self.input_layer[0])],
                                     [np.zeros_like(self.hidden_layers[0]),np.zeros_like(self.hidden_layers[1])],
                                     [np.zeros_like(self.output_layer[0]),np.zeros_like(self.output_layer[1])]]
                    
                    self.second_moment = [[np.zeros_like(self.input_layer[0]),np.zeros_like(self.input_layer[0])],
                                     [np.zeros_like(self.hidden_layers[0]),np.zeros_like(self.hidden_layers[1])],
                                     [np.zeros_like(self.output_layer[0]),np.zeros_like(self.output_layer[1])]]
                self.iter +=1
                beta1 = 0.9
                beta2 = 0.99
                delta = 1e-8
                M = min(5,self.input_size)
                m = int(self.input_size/M)
                for _ in range(m):
                    random_index = M*np.random.randint(0,m)
                    xi = x[random_index:random_index+M]
                    yi = y[random_index:random_index+M]

                    paramsi = [[params[0][0][random_index:random_index+M,:],params[0][1]],
                                params[1],
                                [params[2][0][:,random_index:random_index+M],params[2][1][random_index:random_index+M]]]

                    gradients = grad(self.cost,argnums=0)(paramsi,yi,xi)

                    slicing = [[slice(random_index,random_index+M),slice(None)],
                               [slice(None),slice(None)],
                               [slice(None),slice(random_index,random_index+M)]]
                    
                    for j,s in enumerate(slicing):
                        # Caluclate moments:
                        self.first_moment[j][0][s[0],s[1]] = beta1*self.first_moment[j][0][s[0],s[1]] + (1-beta1)*gradients[j][0]
                        self.second_moment[j][0][s[0],s[1]] = beta2*self.second_moment[j][0][s[0],s[1]] + (1-beta2)*gradients[j][0]*gradients[j][0]
                        
                        # Calculate terms:
                        first_term = self.first_moment[j][0][s[0],s[1]]/(1.0-beta1**self.iter)
                        second_term = self.second_moment[j][0][s[0],s[1]]/(1.0-beta2**self.iter)
                        
                        # Update params:
                        self.params[j][0][s[0],s[1]] -= self.learning_rate*first_term/(np.sqrt(second_term)+delta)

                        if j==2: # this is simply because the output needs slicing.
                            self.first_moment[j][1][s[1]] = beta1*self.first_moment[j][1][s[1]] + (1-beta1)*gradients[j][1]
                            self.second_moment[j][1][s[1]] = beta2*self.second_moment[j][1][s[1]] + (1-beta2)*gradients[j][1]*gradients[j][1]
                            
                            # Calculate terms:
                            first_term = self.first_moment[j][1][s[1]]/(1.0-beta1**self.iter)
                            second_term = self.second_moment[j][1][s[1]]/(1.0-beta2**self.iter)
                            
                            # Update params:
                            self.params[j][1][s[1]] -= self.learning_rate*first_term/(np.sqrt(second_term)+delta)
                        else:
                            self.first_moment[j][1] = beta1*self.first_moment[j][1] + (1-beta1)*gradients[j][1]
                            self.second_moment[j][1] = beta2*self.second_moment[j][1] + (1-beta2)*gradients[j][1]*gradients[j][1]
                            
                            # Calculate terms:
                            first_term = self.first_moment[j][1]/(1.0-beta1**self.iter)
                            second_term = self.second_moment[j][1]/(1.0-beta2**self.iter)
                            
                            # Update params:
                            self.params[j][1] -= self.learning_rate*first_term/(np.sqrt(second_term)+delta)


                self.cost(params,y,x)


    def train(self,x,y,epochs=10,threshold=1e-6):
        '''
        Training method.
        '''
        j = len(x)
        for epoch in range(epochs):
            n = 0
            for xi, yi in zip(x,y):
                self.backprop(yi, xi)
                n+=1
                sys.stdout.write(f"\rProgress: {100*n/j}%, ")
                sys.stdout.flush()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {self.loss:.6f}')
            if self.loss < threshold:
                print('Convergence reached.')
                break
            elif np.isnan(self.loss):
                print('Loss turned nan')
                break

if __name__ == '__main__':
    min_ = 5
    max_ = 50
    input_size_ = 10 #np.random.randint(min, max)
    output_size_ = 5 #np.random.randint(min, max)
    hidden_size_ = 5 #np.random.randint(min, max)
    num_hidden_layers_ = 5 #np.random.randint(min, max)
    data_size = 100

    model = FFNN(hidden_size=hidden_size_,
                num_hidden_layers=num_hidden_layers_,
                input_size=input_size_,
                output_size=output_size_,
                learning_rate=0.001)

    model.set_optimizer('Adam')
    model.set_activation_function(model.sigmoid)
    model.output_function(model.classify_output)

    x_ = np.random.rand(data_size,input_size_)
    y_ = np.random.rand(data_size,output_size_)
    model.train(x=x_, y=y_, epochs=20,threshold=1e-6)
