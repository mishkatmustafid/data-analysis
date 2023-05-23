import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load time series data from a CSV file
data = pd.read_csv('data.csv', parse_dates=['date'], index_col='date')

# Print the first few rows of the data
print(data.head())

# Resample the data to monthly frequency
monthly_data = data.resample('M').mean()

# Visualize the time series data
plt.plot(monthly_data)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series Data')
plt.show()

# Calculate the rolling mean and standard deviation
rolling_mean = monthly_data.rolling(window=12).mean()
rolling_std = monthly_data.rolling(window=12).std()

# Visualize the rolling statistics
plt.plot(monthly_data, color='blue', label='Original')
plt.plot(rolling_mean, color='red', label='Rolling Mean')
plt.plot(rolling_std, color='black', label='Rolling Std')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Rolling Statistics')
plt.legend()
plt.show()

# Perform time series decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(monthly_data)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Visualize the decomposed components
plt.subplot(411)
plt.plot(monthly_data, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
