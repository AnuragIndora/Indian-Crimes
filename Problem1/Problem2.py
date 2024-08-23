import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import numpy as np

# Load your data
data = pd.read_pickle("FeatureCrimeData.pkl")
df = data.copy()

# Convert 'date_reported' to datetime
df['date_reported'] = pd.to_datetime(df['date_reported'])

# Set 'date_reported' as the index
df.set_index('date_reported', inplace=True)

# Aggregate crime counts per month
monthly_crime_counts = df.resample('ME').size()

# Visualize the crime counts
plt.figure(figsize=(14, 7))
plt.plot(monthly_crime_counts, label='Monthly Crime Counts')
plt.title('Monthly Crime Counts Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Crimes')
plt.legend()
plt.show()

# Decompose the time series
result = seasonal_decompose(monthly_crime_counts, model='additive')
result.plot()
plt.show()

# Split data into training and test sets
train_size = int(len(monthly_crime_counts) * 0.8)
train, test = monthly_crime_counts[:train_size], monthly_crime_counts[train_size:]

# Define and fit the ARIMA model
model = SARIMAX(train, 
                 order=(1, 1, 1),  # Adjust the (p,d,q) parameters
                 seasonal_order=(1, 1, 1, 12))  # Adjust (P,D,Q,s) parameters
results = model.fit()

# Forecast
forecast = results.get_forecast(steps=len(test))
forecast_index = pd.date_range(start=test.index[0], periods=len(test), freq='ME')
forecast_values = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# Plot forecasts
plt.figure(figsize=(14, 7))
plt.plot(train, label='Training Data')
plt.plot(test, label='Test Data', color='gray')
plt.plot(forecast_index, forecast_values, label='Forecast', color='red')
plt.fill_between(forecast_index, 
                 confidence_intervals.iloc[:, 0], 
                 confidence_intervals.iloc[:, 1], 
                 color='red', alpha=0.3)
plt.title('Crime Forecast')
plt.xlabel('Date')
plt.ylabel('Number of Crimes')
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(test, forecast_values)
print(f'Mean Squared Error of the Forecast: {mse:.2f}')