'''
A module for doing regression tasks with the help of gradient methods.
Where as for a neural network the weight and biases are the tunable parameters
here the coefficients of the design matrix will be the tunable parameters.
We will use both OLS and Ridge (L2) regression with gradient methods and compare 
them to that of the analytic inversions.
'''
import numpy as np

class Regression:
    '''
    Regression class using analytic inversion or gradient method.
    '''
    def __init__(self,target):
        self.target = target
        self.design_matrix = None
        self.beta = None
        self.lamda = 0.8
        self.loss = None
        self.n=len(target)

    def mse(self,input_,target):
        '''Mean square error.'''
        return np.mean((target-input_).T@(target-input_))

    def ols_grad(self,beta):
        '''Analytic gradient of OLS with respect to MSE.'''
        return 2.0/self.n*self.design_matrix.T @ ((self.design_matrix @ beta)-self.target)

    def ols_inv(self):
        '''Analytic inversion OLS.'''
        return np.linalg.inv(self.design_matrix.T@self.design_matrix) @ (self.design_matrix.T @ self.target)

    def ridge_grad(self,beta):
        '''Analytic gradient of Ridge with respect to MSE.'''
        return (2.0 / self.n) * (self.design_matrix.T @ ((self.design_matrix @ beta) - self.target) + self.lamda * beta)

    def ridge_inv(self):
        '''Analytic inversion Ridge.'''
        I = np.eye(self.design_matrix.shape[1])
        return np.linalg.inv(self.design_matrix.T @ self.design_matrix + self.lamda * I) @ (self.design_matrix.T @ self.target)
    def make_design_matrix(self,x,deg):
        '''Method for generating the design matrix.'''
        mat = np.ones((x.shape[0], deg))
        for j in range(1, deg):
            mat[:, j] = x ** j
        self.design_matrix = mat

    def step(self,beta,grad,lr):
        '''Step using gradient method on the parameters.'''
        return beta - lr * grad

    def train(self,epochs,method='OLS',lr=0.01,threshold=1e-4):
        '''Training function for the gradient methods,
        calls the step method
        '''
        if method not in ['OLS', 'Ridge']:
            raise ValueError("Method must be either 'OLS' or 'Ridge'")
        self.beta = np.zeros(self.design_matrix.shape[1])
        grad_fn = self.ols_grad if method == 'OLS' else self.ridge_grad
        
        for epoch in range(epochs):
            grads = grad_fn(self.beta)
            self.beta = self.step(self.beta, grads, lr)
            loss = self.mse(self.design_matrix @ self.beta, self.target)
            if loss < threshold:
                print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Convergence reached at threshold: {threshold}')
                break
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
        self.loss = loss
        return self.beta, self.loss


if __name__ == '__main__':
    # Set seed for reproducibility:
    np.random.seed(42)

    # Let there be a target data to hit:
    x_ = np.linspace(-3,3,100)#.reshape(1,10,10)
    y_ = 5*x_**2 + 2*x_ + 3+ np.random.randn(*x_.shape) * 0.1  # Target data with noise

    # Make an instance of the model:
    model = Regression(target=y_)

    # Use the model to initiate the design matrix:
    model.make_design_matrix(x_,deg=3) #ChatGPT said we need to use 3

    # Analytic inversion:
    beta_ols_anal_inv = model.ols_inv()
    beta_ridge_anal_inv = model.ridge_inv()
    
    print('Analytic solution (OLS):', beta_ols_anal_inv)
    print('Analytic solution (Ridge):', beta_ridge_anal_inv)

    # Gradient method training:
    model.train(epochs=1000, method='OLS', lr=0.01)
    beta_ols_grad = model.beta
    print('Gradient method solution (OLS):', beta_ols_grad)

    model.train(epochs=1000, method='Ridge', lr=0.01)
    beta_ridge_grad = model.beta
    print('Gradient method solution (Ridge):', beta_ridge_grad)

