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

    def ridge_grad(self):
        '''Analytic gradient of Ridge with respect to MSE.'''

    def ridge_inv(self,x,beta,y):
        '''Analytic inversion Ridge.'''
        return
    
    def make_design_matrix(self,x,deg):
        '''Method for generating the design matrix.'''
        mat = np.ones_like(x)
        if deg!=0:
            for j in range(deg-1):
                mat = np.concatenate((mat,x**(j+1)))
        self.design_matrix = mat

    def step(self):
        '''Step using gradient method on the parameters.'''
        return

    def train(self,epochs,method,threshold=1e-4):
        '''Training function for the gradient methods,
        calls the step method
        '''
        loss = np.zeros_like(epochs)
        loss = None
        for _ in range(epochs):
            loss = self.mse()
            self.step()
            if loss < threshold:
                print('Print desired convergence reached')
                break

        self.loss = loss


if __name__ == '__main__':
    # Set seed for reproducibility:
    np.random.seed(42)

    # Let there be a target date to hit:
    x_ = np.linspace(-3,3,100)#.reshape(1,10,10)
    y_ = 5*x_**2 + 2*x_ + 3

    # Make an instance of the model:
    model = Regression(target=y_)

    # Use the model to initiate the design matrix:
    model.make_design_matrix(x_,deg=2)

    # Analytic inversion:
    beta_ols_anal_inv = model.ols_inv()

    # Loss of gradient method training:
    # loss_ols_grad = model.train(epochs=50, method='GD')
