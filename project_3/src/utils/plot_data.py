import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
sns.set_theme(style="darkgrid", palette="muted")

def plot_stock_prices(data,tickers):
    '''
    Plot the stock prices over time.
    '''
    data_melted = data.melt(id_vars=['Date'], 
                                   value_vars=tickers, 
                                   var_name='Stock', 
                                   value_name='Price')
    plt.figure(figsize=(6, 6))
    sns.lineplot(data=data_melted, x='Date', y='Price', hue='Stock')
    plt.xlabel('Date',fontsize=20)
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
    plt.ylabel('Price',fontsize=20)
    plt.legend(title='Stock Ticker', title_fontsize=20, fontsize=15, loc='upper left')
    dates = data['Date']
    plt.xticks(dates[::20], rotation=90)
    plt.tight_layout()
    plt.savefig('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/plots/stock_plot.pdf')

def plot_h_o(data):
    '''
    Plot the harmonic oscillator over time.
    '''
    plt.figure(figsize=(6,6))
    sns.lineplot(data=data, x='time', y='angle')
    plt.ylabel('Anlge',fontsize=20)
    plt.xlabel('Time',fontsize=20)
    plt.tight_layout()
    plt.savefig('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/plots/harmonic_oscillator.pdf')


if __name__ == '__main__':
    '''
    Dsiplay data sets:
    '''

    data_pth = '/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/stock_data/stock_data_time.csv'
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    print(f'Loading data from: '+ data_pth)
    data = pd.read_csv(data_pth)
    print(f'Plotting...')
    plot_stock_prices(data, tickers)

    data_pth = '/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/harmonic_occilator_data/h_o_data.csv'
    print(f'Loading data from: '+ data_pth)
    df_ho = pd.read_csv(data_pth)
    print(f'Plotting...')
    plot_h_o(df_ho)
