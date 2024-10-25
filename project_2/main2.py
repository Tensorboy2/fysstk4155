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

def accuracy(y_pred, y_test):
    '''acc func'''
    acc = np.sum(y_pred-y_test)/len(y_test)
    return acc
def classification():
    '''Classification function'''
    
    x_train, x_test, y_train, y_test = prepare_data(test_size=0.2)

    x_train = x_train.values[:,:-1] 
    input_size = x_train.shape[1]
    hidden_sizes = [30,30]
    output_size = len(y_train.shape)

    NN_model = NeuralNetwork(input_size=input_size,
                             hidden_sizes=hidden_sizes,
                             output_size=output_size,
                             activiation='leaky_relu',
                             out='sigmoid')
    optimizer = Adam(learning_rate=0.00001,
                    use_momentum=True)
    trainer = Trainer(model=NN_model,
                      optimizer=optimizer,
                      loss_fn=NN_model.cross_entropy_loss)
    y_train = np.where(y_train == 'M', 0, 1)
    print(x_train.shape)
    trainer.train(x_train=x_train,
                  y_train=y_train,
                  epochs=200,
                  batch_size=int(len(y_train)/4))
    
    x_test = x_test.values[:,:-1]
    y_test = np.where(y_test == 'M', 0, 1)
    y_pred = NN_model.forward(NN_model.params,x_test)
    y_pred_train = NN_model.forward(NN_model.params,x_train)

    for pred, target in zip(y_pred_train, y_train):
        print(pred, target)
    for pred, target in zip(y_pred, y_test):
        print(pred, target)
    print(accuracy(y_pred,y_test))
    



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
    hidden_sizes = [20,30,10]
    output_size = 1

    NN_model = NeuralNetwork(input_size=input_size,
                             hidden_sizes=hidden_sizes,
                             output_size=output_size,
                             activiation='relu',
                             out=None)
    
    optimizer = Adam(learning_rate=0.01,
                   use_momentum=False)
    
    trainer = Trainer(NN_model,
                      optimizer,
                      loss_fn=NN_model.mse_loss,)
    trainer.train(points,target,200,batch_size=1000)



if __name__ == '__main__':
    classification()
    # fitting()