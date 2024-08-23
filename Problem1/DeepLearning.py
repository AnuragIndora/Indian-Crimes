import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Load your data
data = pd.read_pickle("FeatureCrimeData.pkl")
df = data.copy()

# Convert 'date_reported' to datetime
df['date_reported'] = pd.to_datetime(df['date_reported'])

# Set 'date_reported' as the index
df.set_index('date_reported', inplace=True)

# Aggregate crime counts per month
monthly_crime_counts = df.resample('ME').size()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(monthly_crime_counts.values.reshape(-1, 1))

# Convert the data to sequences for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 12  # Number of months to consider for predicting the next month
X, y = create_sequences(scaled_data, seq_length)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# Forecast
y_pred = model.predict(X_test)

# Inverse transform the predictions and actual values
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(monthly_crime_counts.index[seq_length+train_size:], y_test_inv, label='Actual')
plt.plot(monthly_crime_counts.index[seq_length+train_size:], y_pred_inv, label='Predicted', color='red')
plt.title('Crime Forecast with LSTM')
plt.xlabel('Date')
plt.ylabel('Number of Crimes')
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(y_test_inv, y_pred_inv)
print(f'Mean Squared Error of the Forecast: {mse:.2f}')
