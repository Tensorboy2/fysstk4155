'''Neural network module'''
import jax.numpy as jp
import numpy as np
# np.random.seed(42)

class NeuralNetwork:
    '''Neural network class'''
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 output_size,
                 activation='sigmoid',
                 out=None,
                 use_l2=False,
                 l2=0.1):
        self.hidden_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.out = out
        self.use_l2 = use_l2
        self.l2 = l2
        self.params = self.initialize_params(input_size, hidden_sizes, output_size)
        
        # Batch normalization
        self.gamma = {}
        self.beta = {}
        for i in range(1, self.hidden_layers + 1):
            self.gamma[f'gamma{i}'] = jp.ones(hidden_sizes[i-1])
            self.beta[f'beta{i}'] = jp.zeros(hidden_sizes[i-1])

    def initialize_params(self, input_size, hidden_sizes, output_size):
        '''Initialize the parameters'''
        params = {}
        all_layers = [input_size] + hidden_sizes + [output_size]

        for i in range(1, len(all_layers)):
            params[f'W{i}'] = np.random.randn(all_layers[i], all_layers[i-1])  # *0.1
            # params[f'b{i}'] = np.random.randn(all_layers[i],1)*0.1
            params[f'b{i}'] = np.zeros((all_layers[i], 1))
        return params

    def activate(self, z, activation):
        '''Activation functions between layers'''
        if activation == 'sigmoid':
            return 1 / (1 + jp.exp(-z))
        elif activation == 'tanh':
            return jp.tanh(z)
        elif activation == 'relu':
            return jp.maximum(0, z)
        elif activation == 'leaky_relu':
            return jp.maximum(0, z) + 1e-2 * jp.minimum(0, z)
        else:
            return z

    def batch_norm(self, x, gamma, beta, eps=1e-5):
        '''Batch normalization function'''
        mu = jp.mean(x, axis=0)
        var = jp.var(x, axis=0)
        x_hat = (x - mu) / jp.sqrt(var + eps)
        return gamma * x_hat + beta

    def forward(self, params, x):
        '''Feed forward method'''
        for i in range(1, len(params) // 2):
            W = params[f'W{i}']
            b = params[f'b{i}']
            x = jp.dot(x, W.T)
            x = x + b.flatten()
            # gamma = self.gamma[f'gamma{i}']
            # beta = self.beta[f'beta{i}']
            # x = self.batch_norm(x, gamma, beta)
            x = self.activate(x, self.activation)
        
        out_index = len(params) // 2
        W = params[f'W{out_index}']
        b = params[f'b{out_index}']
        x = jp.dot(x, W.T)
        x = x + b.flatten()
        x = self.activate(x, self.out)

        return x

    def mse_loss(self, params, x, y):
        '''Mean Squared Error Loss'''
        predictions = self.forward(params, x)
        if self.use_l2:
            return jp.mean((predictions - y) ** 2 + self.l2)
        else:
            return jp.mean((predictions - y) ** 2)

    def cross_entropy_loss(self, params, x, y):
        '''Cross Entropy Loss'''
        predictions = self.forward(params, x)
        if self.use_l2:
            cross_entropy = -jp.mean(y * jp.log(predictions + 1e-9) + (1 - y) * jp.log(1 - predictions + 1e-9))  # normal cross entropy
            l2_penalty = self.l2 * jp.sum(jp.array([jp.sum(w ** 2) for w in params.values() if 'W' in w]))  # Adds L2 based on the sum of the square of the weights
            return cross_entropy + l2_penalty
        else:
            cross_entropy = jp.mean(-y * jp.log(predictions + 1e-9) + (1 - y) * jp.log(1 - predictions + 1e-9))
            return cross_entropy
        
if __name__ == "__main__":
    # Define parameters
    input_size = 30  # Input features
    hidden_sizes = [64, 32]  # Hidden layer sizes
    output_size = 1  # Output size (e.g., for binary classification)
    
    # Create a NeuralNetwork instance
    model = NeuralNetwork(input_size, hidden_sizes, output_size, activation='sigmoid', out='sigmoid', use_l2=True, l2=0.1)
    
    # Generate random input data (10 samples, 30 features)
    X = np.random.rand(10, input_size)
    y = np.random.randint(0, 2, size=(10, 1))  # Example binary target variable (shape should match output)

    # Perform a forward pass
    predictions = model.forward(model.params, X)
    print("Predictions:", predictions)

    # Calculate MSE loss
    loss = model.mse_loss(model.params, X, y)
    print("MSE Loss:", loss)
    
    # Calculate Cross Entropy loss
    loss = model.cross_entropy_loss(model.params, X, y)
    print("Cross Entropy Loss:", loss)
