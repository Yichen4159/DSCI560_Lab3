import pandas as pd
import yfinance as yf
import numpy as np

def fetch_and_preprocess_stock_data(ticker, start_date, end_date):
    """ Fetch stock data and preprocess it """
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)

    #(forward fill then backward fill)
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)

    # Calculate daily returns
    data['Daily Return'] = data['Close'].pct_change()

    return data

def hybrid_sma_ema_signals(data, short_window=20, long_window=50):
    
    # Initialize the signals DataFrame
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0

    # Calculate SMAs and EMAs
    signals['short_sma'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_sma'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    signals['short_ema'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    signals['long_ema'] = data['Close'].ewm(span=long_window, adjust=False).mean()

    # Generate hybrid signals
    # Buy signal (1) if short-term EMA > long-term SMA and short-term SMA > long-term EMA
    signals['signal'][short_window:] = np.where(
        (signals['short_ema'][short_window:] > signals['long_sma'][short_window:]) & 
        (signals['short_sma'][short_window:] > signals['long_ema'][short_window:]), 1.0, 0.0)

    # Sell signal (-1) if short-term EMA < long-term SMA and short-term SMA < long-term EMA
    signals['signal'][short_window:] = np.where(
        (signals['short_ema'][short_window:] < signals['long_sma'][short_window:]) & 
        (signals['short_sma'][short_window:] < signals['long_ema'][short_window:]), -1.0, signals['signal'][short_window:])

    # Generate trading positions
    signals['positions'] = signals['signal'].diff()

    return signals
