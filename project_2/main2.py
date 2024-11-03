'''Main file'''
import numpy as np
np.random.seed(42)
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
from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score

def polynomial(x,y):
    '''make simple polynomial'''
    c = np.random.randn(4)
    target =  c[0] + c[1]*x +c[2]*y + c[3]*x*y
    return target

def regression():
    # Regression
    # Possible parameters to test:
    lrs = [0.1,0.01]
    l2s = [0.1,0.4,0.9]

    # All combinations of optimizers
    optimizers = [GD(), GD(use_momentum=True), SGD(), SGD(use_momentum=True),
                  AdaGrad(), AdaGrad(use_momentum=True),
                  AdaGrad(use_mini_batch=True), AdaGrad(use_mini_batch=True,use_momentum=True),
                  RMSprop(), Adam()]
    
    results_regression_df = pd.DataFrame(columns=["Learning Rate", "L2 Penalty", "Optimizer", "Model"])
    
    n = 40
    x = np.linspace(-1,1,n)
    y = np.linspace(-1,1,n)

    X,Y = np.meshgrid(x,y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    print(len(points))
    target = polynomial(points[:,0],points[:,1])

    x_train, x_test, y_train, y_test = train_test_split(points,target,test_size=0.4)
    

    input_size = x_train.shape[1]  # Input features
    hidden_sizes = [30, 30]  # Hidden layer sizes
    output_size = 1 # output classification
    epochs = 100
    for lr in lrs:
        for l2 in l2s:
            for optim in optimizers:
                # Neural Network
                Neural_Network = NeuralNetwork(input_size=input_size,
                                   hidden_sizes=hidden_sizes,
                                   output_size=output_size,
                                   activation='relu',
                                   out=None,
                                   use_l2=True,
                                   l2=l2) 
                
                optim.learning_rate = lr
                NN_optim = optim
                trainer_NN = Trainer(model=Neural_Network,
                      optimizer=NN_optim,
                      loss_fn=Neural_Network.mse_loss)
                loss_NN = trainer_NN.train(x_train=x_train,
                            y_train=y_train,
                            epochs=epochs,
                            batch_size=int(len(y_train)))
                pred_NN = Neural_Network.forward(Neural_Network.params,x_test)

                lr_info = f"Learning Rate: {lr}"
                l2_info = f"L2 Regularization: {l2}"
                optim_name = type(optim).__name__

                # Extract momentum and mini-batch usage if available
                momentum_info = f"Momentum: {getattr(optim, 'use_momentum', False)}"
                mini_batch_info = f"Mini-Batch: {getattr(optim, 'use_mini_batch', False)}"

                # Loss and accuracy info
                max_loss = np.max(loss_NN)
                min_loss = np.min(loss_NN)
                # acc_info = f"Accuracy: {acc_NN:.4f}"

                # Print the combined information
                print(f"Grid Search Trial - Optimizer: {optim_name}, {lr_info}, {l2_info}")
                print(f"{momentum_info}, {mini_batch_info}")
                print(f"Max Loss: {max_loss:.4f}, Min Loss: {min_loss:.4f}")
                # print(acc_info)
                print("-" * 50)  # Separator for readability
                dfi = pd.DataFrame({
                        "Learning Rate": [lr]*epochs,
                        "L2 Penalty": [l2]*epochs,
                        "Optimizer": [optim_name]*epochs,
                        "Momentum": [momentum_info]*epochs,
                        "Mini_batch": [mini_batch_info]*epochs,
                        "Model": ['Neural_Network']*epochs,
                        "Loss": loss_NN,
                        "Epoch": np.arange(0, epochs)
                    })
                results_regression_df = pd.concat([results_regression_df,dfi], ignore_index=True)


    optimizers = [GD(), GD(use_momentum=True), SGD(), SGD(use_momentum=True),
                  AdaGrad(), AdaGrad(use_momentum=True),
                  AdaGrad(use_mini_batch=True), AdaGrad(use_mini_batch=True,use_momentum=True),
                  RMSprop(), Adam()]
    for lr in lrs:
        for l2 in l2s:
            for optim in optimizers:
                regression = Regression(x=x_train,
                                                 degree=0,
                                                 log=False,
                                                 use_l2=True,
                                                 l2=l2)
                optim.learning_rate = lr
                R_optim = optim
                trainer_R = Trainer(model=regression,
                      optimizer=R_optim,
                      loss_fn=regression.mse_loss)
                loss_R = trainer_R.train(x_train=x_train,
                            y_train=y_train,
                            epochs=epochs,
                            batch_size=int(len(y_train)))
                # pred_LR = Logistic_regression.forward(Logistic_regression.params,x_test)
                # pred_LR = np.where(pred_LR<0.5,0,1)
                # acc_LR = accuracy_score(y_true=y_test,y_pred=pred_LR)
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
                max_loss = np.max(loss_R)
                min_loss = np.min(loss_R)
                # acc_info = f"Accuracy: {acc_LR:.4f}"

                # Print the combined information
                print(f"Grid Search Trial - Optimizer: {optim_name}, {lr_info}, {l2_info}")
                print(f"{momentum_info}, {mini_batch_info}")
                print(f"Max Loss: {max_loss:.4f}, Min Loss: {min_loss:.4f}")
                # print(acc_info)
                print("-" * 50)  # Separator for readability

                
                   
                dfi = pd.DataFrame({
                        "Learning Rate": [lr]*epochs,
                        "L2 Penalty": [l2]*epochs,
                        "Optimizer": [optim_name]*epochs,
                        "Momentum": [momentum_info]*epochs,
                        "Mini_batch": [mini_batch_info]*epochs,
                        "Model": ['Regression']*epochs,
                        "Loss": loss_R,
                        "Epoch": np.arange(0, epochs)
                    })
                results_regression_df = pd.concat([results_regression_df,dfi], ignore_index=True)  
                
    results_regression_df.to_csv("results_regression.csv", index=False)

    


def classify():
    # Possible parameters to test:
    lrs = np.logspace(-3,-8,9)#[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
    l2s = np.linspace(0.1,0.9,9)#[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    # All combinations of optimizers
    # optimizers = [GD(), GD(use_momentum=True), SGD(), SGD(use_momentum=True),
    #               AdaGrad(), AdaGrad(use_momentum=True),
    #               AdaGrad(use_mini_batch=True), AdaGrad(use_mini_batch=True,use_momentum=True),
    #               RMSprop(), Adam()]
    optimizers = [SGD()]
    
    results_NN_classification_df = pd.DataFrame(columns=["Learning Rate", "L2 Penalty", "Optimizer", "Model"])
    # results_classification_LR_df = pd.DataFrame(columns=["Learning Rate", "L2 Penalty", "Optimizer", "Model", "Epoch", "Loss"])

    # Classification:

    x_train, x_test, y_train, y_test = prepare_data(test_size=0.4)
    x_train = x_train.values[:,:-1] 
    x_test = x_test.values[:,:-1] 
    y_train = np.where(y_train == 'M', 1.0, 0.0)
    y_test = np.where(y_test == 'M', 1.0, 0.0)

    input_size = x_train.shape[1]  # Input features
    hidden_sizes = [30, 30]  # Hidden layer sizes
    output_size = 1 # output classification
    epochs = 100
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
                            epochs=epochs,
                            batch_size=int(len(y_train)/8))
                pred_NN = Neural_Network.forward(Neural_Network.params,x_test)
                pred_NN = np.where(pred_NN<0.5,0,1)
                acc_NN = accuracy_score(y_true=y_test,y_pred=pred_NN)

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
                dfi = pd.DataFrame({
                        "Learning Rate": [lr],
                        "L2 Penalty": [l2],
                        "Optimizer": [optim_name],
                        "Momentum": [momentum_info],
                        "Mini_batch": [mini_batch_info],
                        "Model": ['Neural_Network'],
                        "Accuracy": [acc_NN]
                    })
                results_NN_classification_df = pd.concat([results_NN_classification_df,dfi], ignore_index=True)
    results_NN_classification_df.to_csv("results_NN_classification.csv", index=False)
    # optimizers = [GD(), GD(use_momentum=True), SGD(), SGD(use_momentum=True),
    #               AdaGrad(), AdaGrad(use_momentum=True),
    #               AdaGrad(use_mini_batch=True), AdaGrad(use_mini_batch=True,use_momentum=True),
    #               RMSprop(), Adam()]
    # results_R_classification_df = pd.DataFrame(columns=["Learning Rate", "L2 Penalty", "Optimizer", "Momentum", "Mini_batch", "Model", "Accuracy"])
    # optimizers = [SGD()]
    # for lr in lrs:
    #     for l2 in l2s:
    #         for optim in optimizers:
    #             Logistic_regression = Regression(x=x_train,
    #                                              degree=0,
    #                                              log=False,
    #                                              use_l2=True,
    #                                              l2=l2)
    #             optim.learning_rate = lr
    #             LR_optim = optim
    #             trainer_LR = Trainer(model=Logistic_regression,
    #                   optimizer=LR_optim,
    #                   loss_fn=Logistic_regression.cross_entropy_loss)
    #             loss_LR = trainer_LR.train(x_train=x_train,
    #                         y_train=y_train,
    #                         epochs=epochs,
    #                         batch_size=int(len(y_train)/8))
    #             pred_LR = Logistic_regression.forward(Logistic_regression.params,x_test)
    #             pred_LR = np.where(pred_LR<0.5,0,1)
    #             acc_LR = accuracy_score(y_true=y_test,y_pred=pred_LR)
    #             # print(f"Grid Search Trial - Learning Rate: {lr}, L2 Regularization: {l2}, Optimizer: {optim}")
    #             # print(f"Max Loss: {np.max(loss_LR):.4f}, Min Loss: {np.min(loss_LR):.4f}")
    #             # print(f"Accuracy: {acc_LR:.4f}")
    #             lr_info = f"Learning Rate: {lr}"
    #             l2_info = f"L2 Regularization: {l2}"
    #             optim_name = type(optim).__name__

    #             # Extract momentum and mini-batch usage if available
    #             momentum_info = f"Momentum: {getattr(optim, 'use_momentum', False)}"
    #             mini_batch_info = f"Mini-Batch: {getattr(optim, 'use_mini_batch', False)}"

    #             # Loss and accuracy info
    #             max_loss = np.max(loss_LR)
    #             min_loss = np.min(loss_LR)
    #             acc_info = f"Accuracy: {acc_LR:.4f}"

    #             # Print the combined information
    #             print(f"Grid Search Trial - Optimizer: {optim_name}, {lr_info}, {l2_info}")
    #             print(f"{momentum_info}, {mini_batch_info}")
    #             print(f"Max Loss: {max_loss:.4f}, Min Loss: {min_loss:.4f}")
    #             print(acc_info)
    #             print("-" * 50)  # Separator for readability

                
                   
    #             dfi = pd.DataFrame({
    #                     "Learning Rate": [lr],
    #                     "L2 Penalty": [l2],
    #                     "Optimizer": [optim_name],
    #                     "Momentum": [momentum_info],
    #                     "Mini_batch": [mini_batch_info],
    #                     "Model": ['Logistic_regression'],
    #                     "Accuracy": [acc_LR]
    #                 })
    #             results_R_classification_df = pd.concat([results_R_classification_df,dfi], ignore_index=True)  
                
    # # results_classification_NN_df.to_csv("results_classification_NN.csv", index=False)
    # results_R_classification_df.to_csv("results_R_classification.csv", index=False)

if __name__ == '__main__':
    '''
    Data generation for plots:
    '''
    # regression()
    classify()

    