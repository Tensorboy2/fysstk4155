'''Trainer module'''
from jax import grad
import jax.numpy as jp
import sys
import numpy as np
class Trainer:
    '''Trainer class'''
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self,
              x_train,
              y_train,
              epochs,
              batch_size=None,
              x_val=None,
              y_val=None):
        '''Trianing method'''
        params = self.model.params
        
        if (batch_size is None) and self.optimizer.use_mini_batch==False:
            batch_size = len(x_train)

        # num_batches = len(x_train) // batch_size

        for epoch in range(epochs):
            loss = 0
            n = 0
            indices = np.arange(len(x_train))
            np.random.shuffle(indices)
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]

            for i in range(0, len(x_train), batch_size):
                x_batch = x_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]

                
                batch_loss = self.loss_fn(params, x_batch, y_batch)
                grads = grad(self.loss_fn)(params, x_batch, y_batch)

                self.optimizer.step(params, grads)

                loss += batch_loss
                n += len(x_batch)
                sys.stdout.write(f"\rProgress: {100 * n / len(x_train):.0f}%, ")
                sys.stdout.flush()
            if np.isnan(loss):
                print('Loss turned nan. Check lr and grads.')
            if hasattr(self.optimizer,'square_gradeints'):
                self.optimizer.square_gradients=None
            if hasattr(self.optimizer,'first_momentum'):
                self.optimizer.first_momentum=None
                self.optimizer.iter=0
            print(f'\nEpoch: {epoch+1}, avg_loss= {loss/len(y_train):.8f}')


