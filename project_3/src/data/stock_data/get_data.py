import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

'''
Module for getting the stock data with yfinance and scaling usign minmax from sklearn.
'''

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  
start_date = '2023-11-25'
end_date = '2024-11-25'    

data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

scaler = MinMaxScaler(feature_range=(0, 1))

data_scaled = data.copy()
data_scaled[:] = scaler.fit_transform(data)

data_scaled_with_dates = data_scaled.reset_index()
data_scaled_with_dates['Date'] = data_scaled_with_dates['Date'].dt.date
data_with_dates_scaled = data_scaled.copy()
data_with_dates_scaled.to_csv('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/stock_data/stock_data_time.csv', index=True)
