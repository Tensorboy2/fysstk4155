'''Regression module'''
import numpy as np
import jax.numpy as jp
from jax import grad
from sklearn.linear_model import LinearRegression, Ridge as SklearnRidge

class Regression:
    '''Regression method'''
    def __init__(self, x, degree=1,log=False,use_l2=False,l2=0.1):
        self.degree = degree
        self.log = log
        self.use_l2 = use_l2
        self.l2 = l2
        self.X = self.design_matrix(x)
        self.params = {'W1': np.random.randn(self.X.shape[1])*0.001}
        self.target = x
        self.n = len(x)

    def design_matrix(self, x):
        '''Making the design matrix'''
        n_samples, n_features = x.shape
        design_matrix = np.zeros((n_samples, (self.degree+1) * n_features))

        for j in range(self.degree+1):
            design_matrix[:, j*n_features:(j+1)*n_features] = x**j

        return design_matrix
    
    def activation(self,z):
        '''Activation function for classification'''
        if self.log:
            return z
        else:
            return 1/(1+jp.exp(-z))
    
    def forward(self,params,x):
        '''Making prediction from current beta,
        similar to NN forward method in the gradient process.'''
        z = jp.dot(x,params['W1'])
        y = self.activation(z)
        return y

    def mse_loss(self,params,x,y):
        '''Loss'''
        predictions = self.forward(params,x)
        if self.use_l2:
            return jp.mean((predictions- y)**2 + self.l2 * jp.sum(jp.square(params['W1'])))
        else:
            return jp.mean((predictions- y)**2)
    
    def cross_entropy_loss(self,params,x,y):
        '''Cross entropy cost function'''
        predictions = self.forward(params,x)
        if self.use_l2:
            cross_entropy = -jp.mean(y * jp.log(predictions + 1e-9) + (1 - y) * jp.log(1 - predictions + 1e-9))  # Normal cross entropy
            l2_penalty = self.l2 * jp.sum(jp.square(params['W1']))  # Adds L2 based on the sum of the square of the weights
            return cross_entropy + l2_penalty
        else:
            cross_entropy = -jp.mean(y * jp.log(predictions + 1e-9) + (1 - y) * jp.log(1 - predictions + 1e-9))
            return cross_entropy

    def ols_inv(self):
        '''Analytic inversion OLS'''
        return np.linalg.inv(self.X.T @ self.X) @ (self.X.T @ self.target)

    def ridge_inv(self):
        '''Analytic inversion Ridge'''
        I = np.eye(self.X.shape[1])
        return np.linalg.inv(self.X.T @ self.X + self.l2 * I) @ (self.X.T @ self.target)

    def call_ols_grad(self):
        '''Calculate analytical OLS gradient'''
        beta = self.params['W1']
        return 2.0 / self.n * self.X.T @ ((self.X @ beta) - self.target)
    
    def call_ridge_grad(self):
        '''Calculate analytical Ridge gradient'''
        beta = self.params['W1']
        return (2.0 / self.n) * (self.X.T @ ((self.X @ beta) - self.target) + self.l2 * beta)

    def step(self, params, grads, lr):
        '''Update step using gradient method on the parameters'''
        new_params = params.copy()
        new_params['W1'] -= lr * grads
        return new_params

    def train(self, epochs, method='OLS', lr=0.01, threshold=1e-4):
        '''Training function for gradient methods, using the step method'''
        if method not in ['OLS', 'Ridge']:
            raise ValueError("Method must be either 'OLS' or 'Ridge'")
        
        # Select the appropriate gradient function
        grad_fn = self.call_ols_grad if method == 'OLS' else self.call_ridge_grad
        
        for epoch in range(epochs):
            grads = grad_fn()
            self.params = self.step(self.params, grads, lr)
            
            loss = self.mse_loss(self.params, self.X, self.target)
            
            # Early stopping based on threshold
            if loss < threshold:
                print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Convergence reached at threshold: {threshold}')
                break
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
        
        self.loss = loss
        return self.params, self.loss

if __name__ == "__main__":
    np.random.seed(0)
    # Generate random input data (10 samples, 30 features)
    X = np.random.rand(10, 30)
    y = np.random.randint(0, 2, size=(10,))  # Example binary target variable

    # Create a Regression model
    model = Regression(X, degree=2, log=False, use_l2=True, l2=0.1)

    # Example forward pass
    predictions = model.forward(model.params, model.X)
    print("Predictions:", predictions)

    # Example loss calculation
    mse_loss = model.mse_loss(model.params, model.X, y)
    print("MSE Loss:", mse_loss)
    cross_entropy_loss = model.cross_entropy_loss(model.params, model.X, y)
    print("Cross-entropy Loss:", cross_entropy_loss)

    # Analytic Inversion
    beta_ols_anal_inv = model.ols_inv()
    beta_ridge_anal_inv = model.ridge_inv()
    print('Analytic solution (OLS):', beta_ols_anal_inv)
    print('Analytic solution (Ridge):', beta_ridge_anal_inv)
    
    # Train the model using gradient descent
    model.train(epochs=1000, method='OLS', lr=0.01)
    beta_ols_grad = model.params['W1']
    print('Gradient method solution (OLS):', beta_ols_grad)

    model.train(epochs=1000, method='Ridge', lr=0.01)
    beta_ridge_grad = model.params['W1']
    print('Gradient method solution (Ridge):', beta_ridge_grad)

    # Scikit-learn comparison
    lin_reg = LinearRegression().fit(model.X, model.target)
    ridge_reg = SklearnRidge(alpha=model.l2).fit(model.X, model.target)

    print("Scikit-learn OLS coefficients:", lin_reg.coef_)
    print("Scikit-learn Ridge coefficients:", ridge_reg.coef_)

    # Comparison
    print("\nComparison of coefficients:")
    print("Our OLS (analytic):", beta_ols_anal_inv)
    print("Our OLS (grad):", beta_ols_grad)
    print("Sklearn OLS:", lin_reg.coef_)
    
    print("\nOur Ridge (analytic):", beta_ridge_anal_inv)
    print("Our Ridge (grad):", beta_ridge_grad)
    print("Sklearn Ridge:", ridge_reg.coef_)