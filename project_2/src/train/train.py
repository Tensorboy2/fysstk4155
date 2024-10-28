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
              threshold=1e-6,
              batch_size=None,
              x_val=None,
              y_val=None):
        '''Trianing method'''
        params = self.model.params
        
        if (batch_size is None) and self.optimizer.use_mini_batch is False:
            batch_size = len(x_train)

        num_batches = len(x_train) // batch_size
        loss_array = np.zeros(epochs)
        for epoch in range(epochs):
            batch_loss = 0
            indices = np.arange(len(x_train))
            np.random.shuffle(indices)
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]

            for i in range(0, len(x_train), batch_size):
                x_batch = x_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]

                
                batch_lossi = self.loss_fn(params, x_batch, y_batch)
                grads = grad(self.loss_fn,argnums=0)(params, x_batch, y_batch)

                self.optimizer.step(params, grads)

                batch_loss += batch_lossi
            loss = self.loss_fn(params, x_train, y_train)
            loss_array[epoch] = loss
            # print(f'\nEpoch: {epoch+1}, loss = {loss} , avg_batch_loss= {batch_loss/num_batches}')
            if batch_loss/num_batches < threshold:
                print(f'Threshold reached: {threshold} > loss ={batch_loss/num_batches}')
                break

            if np.isnan(batch_loss):
                print('Loss turned nan. Check lr and grads.')
                break
            if hasattr(self.optimizer,'square_gradeints'):
                self.optimizer.square_gradients=None
            if hasattr(self.optimizer,'first_momentum'):
                self.optimizer.first_momentum=None
        return loss_array


