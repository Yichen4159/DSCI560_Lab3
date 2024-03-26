import yfinance as yf
import pandas as pd


def compute_technical_indicators(data):
    # Calculate moving averages
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

    # Compute MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    data['RSI'] = 100 - (100 / (1 + up.rolling(14).mean() / down.abs().rolling(14).mean()))

    # Bollinger Bands
    sma_20 = data['Close'].rolling(window=20).mean()
    rstd_20 = data['Close'].rolling(window=20).std()
    data['Bollinger_Upper'] = sma_20 + 2 * rstd_20
    data['Bollinger_Lower'] = sma_20 - 2 * rstd_20

    # Drop NaN values created by moving averages and indicators
    data = data.dropna()

    return data


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
    data = compute_technical_indicators(data)

    return data


# Example usage
data = fetch_and_preprocess_stock_data("AAPL", "2020-01-01", "2021-12-31")
print(data.tail())
