import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from fetch_for_ARIMA import fetch_and_preprocess_stock_data

# Load data
data = fetch_and_preprocess_stock_data("AAPL", "2020-01-01", "2024-02-01")

# Feature selection
exog_data = data[
    ['SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'RSI', 'Bollinger_Upper', 'Bollinger_Lower']]

# Set up labels
endog_data = data['Close']

# Train test split
train_size = int(len(data) * 0.8)
train_endog = endog_data.iloc[:train_size]
train_exog = exog_data.iloc[:train_size]
test_endog = endog_data.iloc[train_size:]
test_exog = exog_data.iloc[train_size:]

# Using auto model to find the best ARIMA parameter
auto_model = auto_arima(train_endog,
                        exogenous=train_exog,
                        start_p=0, start_q=0,
                        max_p=5, max_q=5,
                        seasonal=False,
                        stepwise=True,
                        suppress_warnings=True,
                        D=1, max_D=1,
                        error_action='ignore')

# Train SARIMAX model with the best parameters
best_order = auto_model.order
best_seasonal_order = auto_model.seasonal_order
model = SARIMAX(train_endog,
                exog=train_exog,
                order=best_order,
                seasonal_order=best_seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False)
model_fit = model.fit()

# Output models
print(f'Best model order: {best_order}')
print(f'Best seasonal order: {best_seasonal_order}')

# Predicting
predictions = model_fit.get_forecast(steps=len(test_endog), exog=test_exog).predicted_mean

# MSE
mse = mean_squared_error(test_endog, predictions)
print(f'Test MSE: {mse}')

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(train_endog.index, train_endog, label='Training Data')
plt.plot(test_endog.index, test_endog, label='Actual Value')
plt.plot(test_endog.index, predictions, label='Predicted Value', color='red')
plt.title('ARIMAX Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# Define signal generator
def generate_signals(predicted, actual, threshold=0.01):
    signals = []
    for pred, act in zip(predicted, actual):
        if pred > act * (1 + threshold):
            signals.append('buy')
        elif pred < act * (1 - threshold):
            signals.append('sell')
        else:
            signals.append('hold')
    return signals


# Generating signals
threshold = 0.01
signals = generate_signals(predictions.values, test_endog.values, threshold)

# Concat signals
test_data_with_signals = test_endog.to_frame(name='Actual')
test_data_with_signals['Predicted'] = predictions
test_data_with_signals['Signal'] = signals

print(test_data_with_signals)
