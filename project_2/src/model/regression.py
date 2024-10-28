'''Regression module'''
import numpy as np
import jax.numpy as jp

class Regression:
    '''Regression method'''
    def __init__(self, x, degree=1,log=False,use_l2=False,l2=0.1):
        self.degree = degree
        self.params = None
        self.log = log
        self.use_l2 = use_l2
        self.l2 = l2
        self.X = self.design_matrix(x)
        self.params = {'W1': np.random.randn(x.shape[1])*0.001}
        # print(f"Number of features (including bias): {self.X.shape[1]}")

    def design_matrix(self,x):
        '''Making the design matrix'''
        # print(f"Input shape: {x.shape}")
        n_samples, n_features = x.shape
        # design_matrix = np.ones((n_samples, 1))  # Bias term
        design_matrix = np.zeros((n_samples,n_features))

        # for i in range(n_features):
        #     for j in range(n_features):
        #         design_matrix[i,j] = 
        # # Generate polynomial features
        # for d in range(1, self.degree + 1):
        #     new_features = []
            # for i in range(n_features):
            #     for j in range(i, n_features):  # To avoid duplicate features (e.g., x1^2, x1*x2, x2^2)
            #         new_feature = (x[:, i] * x[:, j]) ** d
            #         new_features.append(new_feature)

            # design_matrix = np.column_stack((design_matrix, *new_features))
        # print(f"Design matrix shape: {design_matrix.shape}")
        return design_matrix
    
    def activation(self,z):
        '''Activation function for classification'''
        if self.log:
            return z
        else:
            return 1/(1+jp.exp(-z))
    
    def forward(self,params,x):
        '''Making prediction form current beta,
        similar to NN forward method in the gradient process.'''
        # X = self.design_matrix(x)
        # print(f"Input shape: {x.shape}")
        # print()
        # print(f"Number of features (including bias): {params['W1'].shape}")
        z = jp.dot(x,params['W1'])
        # z = np.clip(z,-15,15)
        y = self.activation(z)
        return y

    def mse_loss(self,params,x,y):
        '''Loss'''
        predictions = self.forward(params,x)
        if self.use_l2:
            return jp.mean((predictions- y)**2 + self.l2)
        else:
            return jp.mean((predictions- y)**2)
    
    def cross_entropy_loss(self,params,x,y):
        '''Cross entropy cost function'''
        predictions = self.forward(params,x)
        if self.use_l2:
            cross_entropy = -jp.mean(y * jp.log(predictions + 1e-9) + (1 - y) * jp.log(1 - predictions + 1e-9)) # normal cross entropy
            l2_penalty = self.l2 * jp.sum(jp.array([jp.sum(w**2) for w in params.values() if 'W' in w])) # Adds L2 based on the sum of the square of the weights
            return cross_entropy + l2_penalty
        else:
            cross_entropy = jp.mean(-y * jp.log(predictions + 1e-9) + (1 - y) * jp.log(1 - predictions + 1e-9))
            return cross_entropy


if __name__ == "__main__":
    # Generate random input data (10 samples, 30 features)
    X = np.random.rand(10, 30)
    y = np.random.randint(0, 2, size=(10,))  # Example binary target variable

    # Create a Regression model
    model = Regression(X, degree=0, log=False, use_l2=True, l2=0.1)

    # Example forward pass
    predictions = model.forward(model.params, model.X)
    print("Predictions:", predictions)

    # Example loss calculation
    loss = model.mse_loss(model.params, model.X, y)
    print("MSE Loss:", loss)
    loss = model.cross_entropy_loss(model.params, model.X, y)
    print("Cross entropy Loss:", loss)