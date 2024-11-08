'''Main file'''

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optimi

from src.model.neural_network import NeuralNetwork
from src.model.regression import Regression
from src.model.torch_FFNN import torch_ffnn

from src.optimizer.gd import GD
from src.optimizer.sgd import SGD
from src.optimizer.adagrad import AdaGrad
from src.optimizer.rmsprop import RMSprop
from src.optimizer.adam import Adam
from src.train.train import Trainer
from data.prepare_data import prepare_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

data_path = '/home/sigvar/1_semester/fysstk4155/fysstk_2/project_2/src/utils/'

def polynomial(x, y):
    '''Make simple polynomial'''
    c = np.random.randn(4)
    target = c[0] + c[1]*x + c[2]*y + c[3]*x*y
    return target

def different_activation_functions():
    '''Use different activation functions and save the results to CSV'''

    different_activation_functions_loss_df = pd.DataFrame(
        columns=["Learning Rate", "L2 Penalty", "Optimizer", "Model", "Activation_function", "epochs", "Loss"])

    n = 40
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)

    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    print(len(points))
    target = polynomial(points[:, 0], points[:, 1])

    x_train, x_test, y_train, y_test = train_test_split(points, target, test_size=0.4)

    input_size = x_train.shape[1]  # Input features
    hidden_sizes = [30, 30]  # Hidden layer sizes
    output_size = 1  # Output classification
    epochs = 100
    l2, lr = 0.8, 0.01

    activation_functions = ['relu', 'leaky_relu', 'tanh', 'sigmoid']
    for ac in activation_functions:
        print(f"Grid Search Trial - activation function: {ac}")
        Neural_Network = NeuralNetwork(input_size=input_size,
                                       hidden_sizes=hidden_sizes,
                                       output_size=output_size,
                                       activation=ac,
                                       out=None,
                                       use_l2=True,
                                       l2=l2) 
        optim = SGD(learning_rate=lr)
        trainer_NN = Trainer(model=Neural_Network,
                             optimizer=optim,
                             loss_fn=Neural_Network.mse_loss)
        loss_NN = trainer_NN.train(x_train=x_train,
                                   y_train=y_train,
                                   epochs=epochs,
                                   batch_size=int(len(y_train)/8))
        optim_name = type(optim).__name__
        dfi = pd.DataFrame({
            "Learning Rate": [lr]*epochs,
            "L2 Penalty": [l2]*epochs,
            "Optimizer": [optim_name]*epochs,
            "Model": ['Neural_Network']*epochs,
            "Activation_function": [ac]*epochs,
            "Loss": loss_NN,
            "Epoch": np.arange(0, epochs)
        })
        different_activation_functions_loss_df = pd.concat(
            [different_activation_functions_loss_df, dfi], ignore_index=True)
    different_activation_functions_loss_df.to_csv(
        data_path+"different_activation_functions.csv", index=False)

def regression():
    '''Perform regression with varying learning rates, L2 penalties, and optimizers'''

    lrs = [0.1, 0.01]
    l2s = [0.1, 0.4, 0.9]

    optimizers = [GD(), GD(use_momentum=True), SGD(), SGD(use_momentum=True),
                  AdaGrad(), AdaGrad(use_momentum=True),
                  AdaGrad(use_mini_batch=True), AdaGrad(use_mini_batch=True, use_momentum=True),
                  RMSprop(), Adam()]

    results_regression_df = pd.DataFrame(columns=["Learning Rate", "L2 Penalty", "Optimizer", "Model"])

    n = 40
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)

    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    print(len(points))
    target = polynomial(points[:, 0], points[:, 1])

    x_train, x_test, y_train, y_test = train_test_split(points, target, test_size=0.4)

    input_size = x_train.shape[1]  # Input features
    hidden_sizes = [30, 30]  # Hidden layer sizes
    output_size = 1  # Output classification
    epochs = 100
    batch_size = int(len(y_train)/4)

    # train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    x_train_torch = torch.from_numpy(x_train).float()
    x_test_torch = torch.from_numpy(x_test).float()
    y_train_torch = torch.from_numpy(y_train).float().view(-1, 1)
    y_test_torch = torch.from_numpy(y_test).float().view(-1, 1)

    for lr in lrs:
        for l2 in l2s:
            torch_NN = torch_ffnn(input_size, hidden_sizes, output_size)
            criterion = nn.MSELoss()
            optimizer = optimi.SGD(torch_NN.parameters(), lr=lr, momentum=0.0, weight_decay=l2)
            loss_torch = torch_NN.train_NN(criterion, optimizer, x_train_torch, y_train_torch, epochs)
            max_loss = np.max(loss_torch)
            min_loss = np.min(loss_torch)
            print(f"Grid Search Trial - Optimizer: SGD, {lr}, {l2}")
            print(f"Max Loss: {max_loss:.4f}, Min Loss: {min_loss:.4f}")
            print("-" * 50)  # Separator for readability
            dfi = pd.DataFrame({
                "Learning Rate": [lr]*epochs,
                "L2 Penalty": [l2]*epochs,
                "Optimizer": ['SGD']*epochs,
                "Momentum": [False]*epochs,
                "Mini_batch": [True]*epochs,
                "Model": ['PyTorch']*epochs,
                "Loss": loss_torch,
                "Epoch": np.arange(0, epochs)
            })
            results_regression_df = pd.concat([results_regression_df, dfi], ignore_index=True)

    for lr in lrs:
        for l2 in l2s:
            for optim in optimizers:
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
                                           batch_size=batch_size)
                pred_NN = Neural_Network.forward(Neural_Network.params, x_test)

                lr_info = f"Learning Rate: {lr}"
                l2_info = f"L2 Regularization: {l2}"
                optim_name = type(optim).__name__

                # Extract momentum and mini-batch usage if available
                momentum_info = f"Momentum: {getattr(optim, 'use_momentum', False)}"
                mini_batch_info = f"Mini-Batch: {getattr(optim, 'use_mini_batch', False)}"

                # Loss and accuracy info
                max_loss = np.max(loss_NN)
                min_loss = np.min(loss_NN)

                print(f"Grid Search Trial - Optimizer: {optim_name}, {lr_info}, {l2_info}")
                print(f"{momentum_info}, {mini_batch_info}")
                print(f"Max Loss: {max_loss:.4f}, Min Loss: {min_loss:.4f}")
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
                results_regression_df = pd.concat([results_regression_df, dfi], ignore_index=True)

    optimizers = [GD(), GD(use_momentum=True), SGD(), SGD(use_momentum=True),
                  AdaGrad(), AdaGrad(use_momentum=True),
                  AdaGrad(use_mini_batch=True), AdaGrad(use_mini_batch=True, use_momentum=True),
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
                                         batch_size=batch_size)
                lr_info = f"Learning Rate: {lr}"
                l2_info = f"L2 Regularization: {l2}"
                optim_name = type(optim).__name__

                momentum_info = f"Momentum: {getattr(optim, 'use_momentum', False)}"
                mini_batch_info = f"Mini-Batch: {getattr(optim, 'use_mini_batch', False)}"

                max_loss = np.max(loss_R)
                min_loss = np.min(loss_R)

                print(f"Grid Search Trial - Optimizer: {optim_name}, {lr_info}, {l2_info}")
                print(f"{momentum_info}, {mini_batch_info}")
                print(f"Max Loss: {max_loss:.4f}, Min Loss: {min_loss:.4f}")
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
                results_regression_df = pd.concat([results_regression_df, dfi], ignore_index=True)

    results_regression_df.to_csv(data_path+"results_regression.csv", index=False)

def classify():
    '''Classify using different models and save the results to CSV'''

    resolution = 1

    lrs = np.logspace(-1, -2, resolution)
    l2s = [0.001, 0.01, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    optimizers = [SGD()]

    results_NN_classification_df = pd.DataFrame(columns=["Learning Rate", "L2 Penalty", "Optimizer", "Model"])
    results_torch_classification_df = pd.DataFrame(columns=["Learning Rate", "Optimizer", "Model", "Accuracy"])

    x_train, x_test, y_train, y_test = prepare_data(test_size=0.4)
    y_train = np.where(y_train == 'M', 1.0, 0.0)
    y_test = np.where(y_test == 'M', 1.0, 0.0)

    input_size = x_train.shape[1]  # Input features
    hidden_sizes = [30, 30]  # Hidden layer sizes
    output_size = 1  # Output classification
    epochs = 200
    x_train_torch = torch.from_numpy(x_train).float()
    x_test_torch = torch.from_numpy(x_test).float()
    y_train_torch = torch.from_numpy(y_train).float().view(-1, 1)
    y_test_torch = torch.from_numpy(y_test).float().view(-1, 1)

    for lr in lrs:
        for l2 in l2s:
            torch_NN = torch_ffnn(input_size, hidden_sizes, output_size, classify=True)
            criterion = nn.MSELoss()
            optimizer = optimi.SGD(torch_NN.parameters(), lr=lr, momentum=0.0, weight_decay=l2)

            loss_torch = torch_NN.train_NN(criterion, optimizer, x_train_torch, y_train_torch, epochs)

            with torch.no_grad():
                pred_probs = torch_NN.forward(x_test_torch)
                # pred_probs =  torch.sigmoid(pred_probs) # Apply sigmoid to get probabilities
                pred_torch = (pred_probs >= 0.5).float()
            # pred_torch = torch_NN.forward(x_test_torch).numpy()
            # pred_torch = np.where(pred_torch<0.5,0,1)

            acc_torch = accuracy_score(y_true=y_test_torch.numpy(), y_pred=pred_torch)
            max_loss = np.max(loss_torch)
            min_loss = np.min(loss_torch)
            acc_info = f"Accuracy: {acc_torch:.4f}"
            print(f"Grid Search Trial Torch - Optimizer: SGD, LR:{lr}, L2:{l2}")
            print(f"Max Loss: {max_loss:.4f}, Min Loss: {min_loss:.4f}")
            print(acc_info)
            print("-" * 50)  # Separator for readability

            dfi = pd.DataFrame({
                "Learning Rate": [lr],
                "L2 Penalty": [l2],
                "Optimizer": ['SGD'],
                "Model": ['PyTorch'],
                "Accuracy": [acc_torch]
            })
            results_torch_classification_df = pd.concat([results_torch_classification_df, dfi], ignore_index=True)
    results_torch_classification_df.to_csv(data_path+"results_torch_classification.csv", index=False)

    for lr in lrs:
        for l2 in l2s:
            for optim in optimizers:
                Neural_Network = NeuralNetwork(input_size=input_size,
                                               hidden_sizes=hidden_sizes,
                                               output_size=output_size,
                                               activation='sigmoid',
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
                                           batch_size=int(len(y_train)/4))
                pred_NN = Neural_Network.forward(Neural_Network.params, x_test)
                pred_NN = np.where(pred_NN < 0.5, 0, 1)
                acc_NN = accuracy_score(y_true=y_test, y_pred=pred_NN)

                lr_info = f"Learning Rate: {lr}"
                l2_info = f"L2 Regularization: {l2}"
                optim_name = type(optim).__name__

                # Extract momentum and mini-batch usage if available
                momentum_info = f"Momentum: {getattr(optim, 'use_momentum', False)}"
                mini_batch_info = f"Mini-Batch: {getattr(optim, 'use_mini_batch', False)}"

                max_loss = np.max(loss_NN)
                min_loss = np.min(loss_NN)
                acc_info = f"Accuracy: {acc_NN:.4f}"

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
                results_NN_classification_df = pd.concat([results_NN_classification_df, dfi], ignore_index=True)
    results_NN_classification_df.to_csv(data_path+"results_NN_RMS_classification.csv", index=False)



    # optimizers = [GD(), GD(use_momentum=True), SGD(), SGD(use_momentum=True),
    #               AdaGrad(), AdaGrad(use_momentum=True),
    #               AdaGrad(use_mini_batch=True), AdaGrad(use_mini_batch=True,use_momentum=True),
    #               RMSprop(), Adam()]

    results_R_classification_df = pd.DataFrame(columns=["Learning Rate", "L2 Penalty", "Optimizer", "Momentum", "Mini_batch", "Model", "Accuracy"])

    # Analytic inversion
    # OLS:
    Logistic_regression = Regression(x=x_train,
                                     degree=0,
                                     log=False,
                                     use_l2=False)
    a_beta = Logistic_regression.ols_inv(x_train, y_train)
    y = x_test @ a_beta
    pred_analytic = np.where(y < 0.5, 0, 1)
    print(f'Accuracy for OLS inversion:{accuracy_score(y_true=y_test, y_pred=pred_analytic)}')

    # Ridge regression
    for l2 in l2s:
        Logistic_regression = Regression(x=x_train,
                                         degree=0,
                                         log=False,
                                         use_l2=True,
                                         l2=l2)
        a_ridge_beta = Logistic_regression.ridge_inv(x_train, y_train, l2)
        y = x_test @ a_ridge_beta
        pred_analytic = np.where(y < 0.5, 0, 1)
        print(f'{l2} : {accuracy_score(y_true=y_test, y_pred=pred_analytic)}')

    # Logistic regression
    optimizers = [SGD()]
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
                                           epochs=epochs,
                                           batch_size=int(len(y_train)/16))
                pred_LR = Logistic_regression.forward(Logistic_regression.params, x_test)
                pred_LR = np.where(pred_LR < 0.5, 0, 1)

                acc_LR = accuracy_score(y_true=y_test, y_pred=pred_LR)
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

                
                   
                dfi = pd.DataFrame({
                        "Learning Rate": [lr],
                        "L2 Penalty": [l2],
                        "Optimizer": [optim_name],
                        "Momentum": [momentum_info],
                        "Mini_batch": [mini_batch_info],
                        "Model": ['Logistic_regression'],
                        "Accuracy": [acc_LR]
                    })
                results_R_classification_df = pd.concat([results_R_classification_df,dfi], ignore_index=True)  
                
    # results_classification_NN_df.to_csv("results_classification_NN.csv", index=False)
    results_R_classification_df.to_csv(data_path+"results_R_classification.csv", index=False)

if __name__ == '__main__':
    '''
    Data generation for plots:
    '''
    # different_activation_functions()
    # regression()
    classify()