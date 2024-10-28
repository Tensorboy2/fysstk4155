'''Main file'''
import numpy as np
import pandas as pd

from src.model.neural_network import NeuralNetwork
from src.model.regression import Regression

from src.optimizer.gd import GD
from src.optimizer.sgd import SGD
from src.optimizer.adagrad import AdaGrad
from src.optimizer.rmsprop import RMSprop
from src.optimizer.adam import Adam
from src.train.train import Trainer
from data.prepare_data import prepare_data

from sklearn.metrics import accuracy_score

def accuracy(y_pred, y_test):
    '''acc func'''
    acc = 0
    for pred, t in zip(y_pred, y_test):
        if np.abs(pred-t)<=0.01:
            acc+=1
    return acc/len(y_test)
def classification(lr, lam, optim):
    '''Classification function'''
    
    x_train, x_test, y_train, y_test = prepare_data(test_size=0.2)

    x_train = x_train.values[:,:-1] 
    input_size = x_train.shape[1]
    hidden_sizes = [20]
    output_size = len(y_train.shape)

    NN_model = NeuralNetwork(input_size=input_size,
                             hidden_sizes=hidden_sizes,
                             output_size=output_size,
                             activiation='leaky_relu',
                             out='sigmoid')
    optimizer = Adam(learning_rate=lr,#0.0001,
                    use_momentum=True)
    trainer = Trainer(model=NN_model,
                      optimizer=optimizer,
                      loss_fn=NN_model.cross_entropy_loss)
    y_train = np.where(y_train == 'M', 0.0, 1.0)
    loss = trainer.train(x_train=x_train,
                  y_train=y_train,
                  epochs=100,
                  batch_size=int(len(y_train)/4))
    
    x_test = x_test.values[:,:-1]
    y_test = np.where(y_test == 'M', 0, 1)
    y_pred = NN_model.forward(NN_model.params,x_test)
    y_pred = np.where(y_pred<0.5, 0,1)
    # y_pred_train = NN_model.forward(NN_model.params,x_train)
    acc = accuracy_score(y_test,y_pred.flatten())
    print(acc)
    return acc, loss

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
    # Possible parameters to test:
    lrs = [0.0001,0.00001,0.000001]
    l2s = np.linspace(0.7,0.9,3)

    # All combinations of optimizers
    optimizers = [GD(), GD(use_momentum=True), SGD(), SGD(use_momentum=True),
                  AdaGrad(), AdaGrad(use_momentum=True),
                  AdaGrad(use_mini_batch=True), AdaGrad(use_mini_batch=True,use_momentum=True),
                  RMSprop(), Adam()]
    
    results_classification_df = pd.DataFrame(columns=["Learning Rate", "L2 Penalty", "Optimizer", "Model", "Epoch", "Loss"])

    # Classification:

    x_train, x_test, y_train, y_test = prepare_data(test_size=0.4)
    x_train = x_train.values[:,:-1] 
    x_test = x_test.values[:,:-1] 
    y_train = np.where(y_train == 'M', 1.0, 0.0)
    y_test = np.where(y_test == 'M', 1.0, 0.0)

    input_size = x_train.shape[1]  # Input features
    hidden_sizes = [30, 30]  # Hidden layer sizes
    output_size = 1 # output classification

    for lr in lrs:
        for l2 in l2s:
            for optim in optimizers:
                # Neural Network
                Neural_Network = NeuralNetwork(input_size=input_size,
                                   hidden_sizes=hidden_sizes,
                                   output_size=output_size,
                                   activation='relu',
                                   out='sigmoid',
                                   use_l2=True,
                                   l2=l2) 
                
                optim.learning_rate = lr
                NN_optim = optim
                trainer_NN = Trainer(model=Neural_Network,
                      optimizer=NN_optim,
                      loss_fn=Neural_Network.cross_entropy_loss)
                loss_NN = trainer_NN.train(x_train=x_train,
                            y_train=y_train,
                            epochs=50,
                            batch_size=int(len(y_train)/4))
                pred_NN = Neural_Network.forward(Neural_Network.params,x_test)
                pred_NN = np.where(pred_NN<0.5,0,1)
                acc_NN = accuracy_score(y_true=y_test,y_pred=pred_NN)
                # print(f"Grid Search Trial - Learning Rate: {lr}, L2 Regularization: {l2}, Optimizer: {optim}")
                # print(f"Max Loss: {np.max(loss_NN):.4f}, Min Loss: {np.min(loss_NN):.4f}")
                # print(f"Accuracy: {acc_NN:.4f}")
                lr_info = f"Learning Rate: {lr}"
                l2_info = f"L2 Regularization: {l2}"
                optim_name = type(optim).__name__

                # Extract momentum and mini-batch usage if available
                momentum_info = f"Momentum: {getattr(optim, 'use_momentum', False)}"
                mini_batch_info = f"Mini-Batch: {getattr(optim, 'use_mini_batch', False)}"

                # Loss and accuracy info
                max_loss = np.max(loss_NN)
                min_loss = np.min(loss_NN)
                acc_info = f"Accuracy: {acc_NN:.4f}"

                # Print the combined information
                print(f"Grid Search Trial - Optimizer: {optim_name}, {lr_info}, {l2_info}")
                print(f"{momentum_info}, {mini_batch_info}")
                print(f"Max Loss: {max_loss:.4f}, Min Loss: {min_loss:.4f}")
                print(acc_info)
                print("-" * 50)  # Separator for readability

    optimizers = [GD(), GD(use_momentum=True), SGD(), SGD(use_momentum=True),
                  AdaGrad(), AdaGrad(use_momentum=True),
                  AdaGrad(use_mini_batch=True), AdaGrad(use_mini_batch=True,use_momentum=True),
                  RMSprop(), Adam()]
    for lr in lrs:
        for l2 in l2s:
            for optim in optimizers:
                Logistic_regression = Regression(x=x_train,
                                                 degree=0,
                                                 log=False,
                                                 use_l2=True,
                                                 l2=l2)
                optim.learning_rate = lr
                LR_optim = optim
                trainer_LR = Trainer(model=Logistic_regression,
                      optimizer=LR_optim,
                      loss_fn=Logistic_regression.cross_entropy_loss)
                loss_LR = trainer_LR.train(x_train=x_train,
                            y_train=y_train,
                            epochs=50,
                            batch_size=int(len(y_train)/4))
                pred_LR = Logistic_regression.forward(Logistic_regression.params,x_test)
                pred_LR = np.where(pred_LR<0.5,0,1)
                acc_LR = accuracy_score(y_true=y_test,y_pred=pred_LR)
                # print(f"Grid Search Trial - Learning Rate: {lr}, L2 Regularization: {l2}, Optimizer: {optim}")
                # print(f"Max Loss: {np.max(loss_LR):.4f}, Min Loss: {np.min(loss_LR):.4f}")
                # print(f"Accuracy: {acc_LR:.4f}")
                lr_info = f"Learning Rate: {lr}"
                l2_info = f"L2 Regularization: {l2}"
                optim_name = type(optim).__name__

                # Extract momentum and mini-batch usage if available
                momentum_info = f"Momentum: {getattr(optim, 'use_momentum', False)}"
                mini_batch_info = f"Mini-Batch: {getattr(optim, 'use_mini_batch', False)}"

                # Loss and accuracy info
                max_loss = np.max(loss_LR)
                min_loss = np.min(loss_LR)
                acc_info = f"Accuracy: {acc_LR:.4f}"

                # Print the combined information
                print(f"Grid Search Trial - Optimizer: {optim_name}, {lr_info}, {l2_info}")
                print(f"{momentum_info}, {mini_batch_info}")
                print(f"Max Loss: {max_loss:.4f}, Min Loss: {min_loss:.4f}")
                print(acc_info)
                print("-" * 50)  # Separator for readability

    #             results_classification_df = results_classification_df.append({
    #                     "Learning Rate": lr,
    #                     "L2 Penalty": l2,
    #                     "Optimizer": optim,
    #                     "Model": Neural_Network,
    #                     "Accuracy": acc_NN
    #                 }, ignore_index=True)
                   
    #             results_classification_df = results_classification_df.append({
    #                     "Learning Rate": lr,
    #                     "L2 Penalty": l2,
    #                     "Optimizer": optim,
    #                     "Model": Logistic_regression,
    #                     "Accuracy": acc_LR
    #                 }, ignore_index=True)   
    # results_classification_df.to_csv("results_classification.csv", index=False)