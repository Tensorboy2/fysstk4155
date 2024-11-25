import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler



# Define stock tickers and the time span
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Example stocks
start_date = '2023-11-25'  # One year ago
end_date = '2024-11-25'    # Today

# Fetch stock data
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Reset index to include dates as a column
data_with_dates = data.reset_index()

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale all stock columns
data_scaled = data.copy()  # Create a copy to preserve original
data_scaled[:] = scaler.fit_transform(data)  # Scale all columns

# Save scaled data with dates to CSV
data_with_dates_scaled = data_scaled.copy()
# data_with_dates_scaled.insert(0, 'Date', data_with_dates['Date'])
data_with_dates_scaled.to_csv('/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/stock_data_time.csv', index=False)

print(data_with_dates_scaled.head())
