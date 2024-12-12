import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy, norm

def plot_average_step_differences_histogram(df, bins=50, fit_distributions=True):
    """
    Plots a histogram of the average step differences across all companies in the stock data.
    """
    if 'Date' in df.columns:
        df = df.set_index('Date')
    
    df = df.sort_index()
    
    step_differences = df.diff().iloc[1:]
    
    average_step_differences = step_differences.mean(axis=1).values
    
    plt.figure(figsize=(6, 6))
    plt.hist(average_step_differences, bins=bins, density=True, alpha=0.6, color='blue', label='Average Step Differences')
    
    if fit_distributions:
        x = np.linspace(min(average_step_differences), max(average_step_differences), 1000)
        
        cauchy_params = cauchy.fit(average_step_differences)
        plt.plot(x, cauchy.pdf(x, *cauchy_params), 'r--', label='Cauchy Fit')
        
        mean, std = np.mean(average_step_differences), np.std(average_step_differences)
        plt.plot(x, norm.pdf(x, mean, std), 'g--', label='Normal Fit')
    
    plt.xlabel('Average Difference',fontsize=20)
    plt.ylabel('Density',fontsize=20)
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/plots/step_distribution_stocks.pdf')


data = pd.read_csv('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/stock_data/stock_data_time.csv')
data['Date']= pd.to_datetime(data['Date'])

plot_average_step_differences_histogram(data)

def plot_step_differences_histogram(df):
    """
    Plots a histogram of the step difference in the harmonic oscillator data.
    """
    step_differences = np.diff(df[df.columns[2]].values)
    plt.figure(figsize=(6, 6))
    plt.hist(step_differences, bins=50, density=True, alpha=0.6, color='blue', label='Step Differences')
    x = np.linspace(min(step_differences), max(step_differences), 1000)
    cauchy_params = cauchy.fit(step_differences)
    plt.plot(x, cauchy.pdf(x, *cauchy_params), 'r--', label='Cauchy Fit')
    mean, std = np.mean(step_differences), np.std(step_differences)
    plt.plot(x, norm.pdf(x, mean, std), 'g--', label='Normal Fit')
    plt.xlabel('Difference',fontsize=20)
    plt.ylabel('Density',fontsize=20)
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/plots/step_distribution_ho.pdf')


df = pd.read_csv('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/harmonic_occilator_data/h_o_data.csv') 
plot_step_differences_histogram(df)
