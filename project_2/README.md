# Project 2 for FYSSTK 4155 Fall 2024
## **by Sigurd Vargdal and Michele Tognoni**

### Title:
## Classification and Regression: From Linear and Logistic Regression to Neural Networks

This repository contains the source code and materials for Project 2 of the FYSSTK 4155 course. The project focuses on various classification and regression techniques, ranging from traditional linear and logistic regression methods to more complex neural networks. The repository is organized into multiple directories, each serving a particular component of the project.

- The `model` directory contains implementations of loss functions, neural network architecture, and regression models.
- The `optimizer` directory includes various optimization algorithms used for training the models.
- The `train` directory holds the training loop and associated functions.
- The `utils` directory consists of utility functions for evaluation and plotting.

The `main2` file allows you to manually set hyperparameters, such as sizes and other parameters, and generates CSV files that can be used with the `plotting.py` script to produce visualizations.

Below is a detailed structure and explanation of the repository:

### Repository Structure:
```plaintext
src/
├── model/
│   ├── LossFunctions/
│   │   ├── **Mean Squared Error (MSE)**: This module is used for regression tasks to minimize the squared difference between predicted and actual values.
│   │   └── **Cross Entropy**: This module is used for classification tasks to measure the difference between predicted probabilities and actual classes.
│   ├── NeuralNetwork/
│   │   ├── **Layers**: 
│   │   │   ├── Input Layer: This layer takes the shape of the input data.
│   │   │   ├── Hidden Layers: This section allows configuring a number of hidden layers and neurons per layer based on the task.
│   │   │   └── Output Layer: This layer is configurable based on the task, whether it's regression or classification.
│   │   ├── **Activation Functions**: 
│   │   │   ├── Common Activations: Includes ReLU, Sigmoid, Tanh which are typically used in hidden layers.
│   │   │   └── Optional Activation: For the output layer, like Softmax for classification tasks.
│   │   ├── **Weight and Bias Storage**: Weights and biases are stored in a dictionary format under `self.params`.
│   │   └── **Forward Propagation**: Implements the forward pass through the neural network layers with activation functions applied.
│   └── Regression/
│       ├── **Design Matrix Generation**: This module creates a design matrix `X` based on the data shape and polynomial fit degree for linear or polynomial regression tasks.
│       └── **Parameter Storage**: This stores the regression coefficients in `self.params`, which can be obtained using methods like Ordinary Least Squares (OLS) or Ridge regression.

├── optimizer/
│   ├── **Base Optimizer**:
│   │   ├── Common Attributes: Includes shared attributes like learning rate.
│   │   └── **Step Function**: An interface that each specific optimizer implements to update weights.
│   └── Optimizers/
│       ├── **Gradient Descent (GD)**: Basic gradient descent algorithm.
│       ├── **Stochastic Gradient Descent (SGD)**: Optimizer that uses mini-batches for updating weights.
│       ├── **AdaGrad**: Adaptation of gradient descent that can work with both GD and SGD using mini-batches.
│       ├── **RMSprop**: Optimizer that uses mini-batches for updating weights.
│       ├── **Adam**: Advanced optimizer that includes momentum and adaptive learning rates.

├── train/
│   └── **Training Function**:
│       ├── **Training Loop**: 
│       │   ├── Iterates through the dataset for a number of epochs.
│       │   ├── At each epoch, calls `optimizer.step()` to update weights.
│       │   └── Evaluates model on validation data during the training process.
│       ├── **Loss Tracking**: Logs and stores loss values over epochs to monitor model convergence.
│       └── **Early Stopping**: An optional feature to halt training when performance stops improving.

└── utils/
    ├── EvaluationFunctions/
    │   ├── **Mean Squared Error (MSE)**: This function measures the average squared difference between predicted and actual values, mainly for regression tasks.
    │   └── **R2 Score**: This function evaluates the proportion of variance explained by the model in regression tasks.
    └── Plotting/
        ├── **Accuracy Plots**: Visualize accuracy or performance as a function of different hyperparameters (e.g., learning rate, batch size).
        ├── **Loss over Epochs**: Plots loss values during the training process to visualize model convergence.
        └── **Classification Coefficients**: For linear models, this allows visualization of the learned coefficients.
