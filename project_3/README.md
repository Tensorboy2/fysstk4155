# Project 3 Fys-Stk4155
### By Sigurd Vargdal

## Description
Code for project 3 in Fys-Stk4155. This project was an analysis of the predictive power of the recurrent neural network compared to the standard feed forward neural network.

## Structure
- src/
  - data/
    - harmonic_osscilator_data/
        - generate_data.py: Generates the data for the angle of a pendulum over time using a Runge Kutta solver.
    - stock_data/
        - get_data.py: Fetches the stock data trough the yfinance pip package.
  - model/
    - FFNN.py: Defines the feedforward neural network architecture and training process.
    - RNN.py: Implements the recurrent neural network and training process.
  - utils/
    - make_dist_plot.py: Generates distribution plots for the input data.
    - plot_data.py: Visualizes input data.
