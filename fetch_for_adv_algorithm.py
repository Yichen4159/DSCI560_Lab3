import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def compute_RSI(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_MACD(data, n_fast=12, n_slow=26):
    ema_fast = data['Close'].ewm(span=n_fast, min_periods=n_fast).mean()
    ema_slow = data['Close'].ewm(span=n_slow, min_periods=n_slow).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, min_periods=9).mean()
    return macd, signal


def fetch_and_preprocess_stock_data(ticker, start_date, end_date):
    """Fetch stock data, preprocess it, and calculate technical indicators"""
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)

    # Fill missing values
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)

    # Calculate daily returns
    data['Daily Return'] = data['Close'].pct_change()

    # Calculate technical indicators
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = compute_RSI(data['Close'], 14)
    macd, signal = compute_MACD(data)
    data['MACD'] = macd
    data['MACD_Signal'] = signal

    # Standardize the data
    features_to_scale = ['Close', 'Volume', 'MA50', 'MA200', 'RSI', 'MACD', 'MACD_Signal']
    scaler = StandardScaler()
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

    # Drop rows with NaN values created by rolling windows
    data.dropna(inplace=True)

    return data


# Example usage
data = fetch_and_preprocess_stock_data("AAPL", "2020-01-01", "2023-12-31")
print(data.head())
