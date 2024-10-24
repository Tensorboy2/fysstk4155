'''Main file'''
import numpy as np

from src.model.neural_network import NeuralNetwork
from src.optimizer.gd import GD
from src.optimizer.sgd import SGD
from src.optimizer.adagrad import AdaGrad
from src.optimizer.rmsprop import RMSprop
from src.optimizer.adam import Adam
from src.train.train import Trainer
from data.prepare_data import prepare_data

def classification():
    '''Classification function'''
    
    x_train, x_test, y_train, y_test = prepare_data(test_size=0.8)

    x_train = x_train.values[:,:-1]
    input_size = x_train.shape[1]
    hidden_sizes = [30]
    output_size = len(y_train.shape)

    NN_model = NeuralNetwork(input_size=input_size,
                             hidden_sizes=hidden_sizes,
                             output_size=output_size,
                             activiation='tanh',
                             out='sigmoid')
    optimizer = Adam(learning_rate=0.01,
                    use_momentum=False)
    trainer = Trainer(model=NN_model,
                      optimizer=optimizer,
                      loss_fn=NN_model.cross_entropy_loss)
    y_targets = np.where(y_train == 'M', 1, 0)
    print(x_train.shape)
    trainer.train(x_train=x_train,
                  y_train=y_targets,
                  epochs=200,
                  batch_size=455)
    



def polynomial(x,y):
    a = np.random.randn()
    target =  100 + a*x +a*5*y + a*2*x*y
    return target

def fitting():
    n = 40
    x = np.linspace(-1,1,n)
    y = np.linspace(-1,1,n)

    X,Y = np.meshgrid(x,y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    print(len(points))
    target = polynomial(points[:,0],points[:,1])
    

    input_size = 2
    hidden_sizes = [200,300,100]
    output_size = 1

    NN_model = NeuralNetwork(input_size=input_size,
                             hidden_sizes=hidden_sizes,
                             output_size=output_size,
                             activiation='sigmoid',
                             out=None)
    
    optimizer = Adam(learning_rate=0.01,
                   use_momentum=False)
    
    trainer = Trainer(NN_model,
                      optimizer,
                      loss_fn=NN_model.mse_loss,)
    trainer.train(points,target,100,batch_size=1000)



if __name__ == '__main__':
    classification()
    # fitting()