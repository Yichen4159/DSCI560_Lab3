import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
import matplotlib.pyplot as plt
from fetch_for_adv_algorithm import fetch_and_preprocess_stock_data

# Load Data
data = fetch_and_preprocess_stock_data("AAPL", "2020-01-01", "2024-02-01")

# Data pre-processing
close_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Define time window
time_step = 60
X, y = [], []
for i in range(len(scaled_data) - time_step - 1):
    X.append(scaled_data[i:(i + time_step), 0])
    y.append(scaled_data[i + time_step, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM building
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Set up early stop point
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Set up check point
model_checkpoint = ModelCheckpoint('lstm_model.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)

# Start Training
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop, model_checkpoint]
)

# Load Weights
model.load_weights('lstm_model.h5')

# Test Training
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

# Evaluation
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = MeanAbsoluteError()
rmse = RootMeanSquaredError()
mae.update_state(y_test, predictions)
rmse.update_state(y_test, predictions)

print(f"MAE: {mae.result().numpy()}, RMSE: {rmse.result().numpy()}")

plt.figure(figsize=(10,6))
plt.plot(y_test, color='blue', label='Actual Stock Price')
plt.plot(predictions , color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Last price
last_actual_price = data['Close'].iloc[-1]

predicted_prices = predictions[:, -1]

# Generate Signals
signals = ['Buy' if pred > last_actual_price else 'Sell' for pred in predicted_prices]
signals_series = pd.Series(signals, index=data.index[-len(signals):])

print(signals_series)

