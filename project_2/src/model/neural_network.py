'''Neural network module'''
import jax.numpy as jp
import numpy as np
np.random.seed(42)

class NeuralNetwork:
    '''Neural network'''
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 output_size,
                 activiation = 'sigmoid',
                 out = None):
        self.hidden_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.activation = activiation
        self.out = out
        self.params = self.initialize_params(input_size,
                                             hidden_sizes,
                                             output_size)
        self.gamma = {}  # Scaling factor
        self.beta = {}   # Shifting factor
        # Initialize gamma and beta to 1 and 0, respectively, for each layer where batch normalization is applied
        for i in range(1, self.hidden_layers+1):
            self.gamma[f'gamma{i}'] = jp.ones(hidden_sizes[i-1])
            self.beta[f'beta{i}'] = jp.zeros(hidden_sizes[i-1])

    def initialize_params(self,
                          input_size,
                          hidden_sizes,
                          output_size):
        '''Initialize the parametes'''
        params = {}
        all_layers = [input_size] + hidden_sizes + [output_size]
        # print(len(all_layers))

        for i in range(1,len(all_layers)):
            # print(i)
            params[f'W{i}'] = np.random.randn(all_layers[i],all_layers[i-1])#*0.1
            params[f'b{i}'] = np.random.randn(all_layers[i],1)#*0.1
            # params[f'b{i}'] = np.zeros((all_layers[i],1))
        # print('1 hei')
        return params

    def activate(self,z,activation):
        '''Activation functions between layers'''
        if activation == 'sigmoid':
            return 1/(1+jp.exp(-z))
        elif activation == 'tanh':
            return jp.tanh(z)
        elif activation == 'relu':
            return jp.maximum(0,z)
        elif activation == 'leaky_relu':
            return jp.maximum(0,z) + 1e-2*jp.minimum(0,z)
        else:
            return z

    def batch_norm(self, x, gamma ,beta , eps=1e-5):
        '''Batch normalization function.'''
        mu = jp.mean(x, axis=0)
        var = jp.var(x, axis=0)
        x_hat = (x - mu) / jp.sqrt(var + eps)
        return gamma * x_hat + beta

    # def soft_max(self,z):
    #     '''Soft max function for classification problems'''
    #     exp_z = jp.exp(z-jp.max(z,axis=0,keepdims=True))
    #     return exp_z / jp.sum(exp_z,axis=0,keepdims=True)

    def forward(self,params,x):
        '''Feed forward method'''
        # print(len(params))
        for i in range(1, len(params)//2 ):
            W = params[f'W{i}']
            b = params[f'b{i}']
            x = jp.dot(x,W.T)
            x = x+b.flatten()
            gamma = self.gamma[f'gamma{i}']
            beta = self.beta[f'beta{i}']
            x = self.batch_norm(x, gamma, beta)
            x = self.activate(x,self.activation)
        
        out_index = len(params)//2
        W = params[f'W{out_index}']
        b = params[f'b{out_index}']
        x = jp.dot(x,W.T)
        x = x+b.flatten()
        # for i in range(10):
        #     x = self.activate(x,'sigmoid')
        x = self.activate(x,self.out)

        return x

    def mse_loss(self,params,x,y):
        '''Loss'''
        predictions = self.forward(params,x)
        # print(type(predictions))
        # print('hei')
        # print(predictions.shape)
        # print(y.shape)
        return jp.mean((predictions- y)**2 + 0.1)

    def cross_entropy_loss(self,params,x,y, l2_lambda=0.9):
        '''Cross entropy cost function'''
        predictions = self.forward(params,x)
        cross_entropy = -jp.mean(y * jp.log(predictions + 1e-9))
        return cross_entropy
        # print(predictions)
        # cross_entropy = -jp.mean(y * jp.log(predictions + 1e-9) + (1 - y) * jp.log(1 - predictions + 1e-9))
        # l2_penalty = l2_lambda * jp.sum(jp.array([jp.sum(w**2) for w in params.values() if 'W' in w]))
        # return cross_entropy + l2_penalty