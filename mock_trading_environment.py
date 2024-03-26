import yfinance as yf
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pmdarima import auto_arima



portfolio_values = {}
portfolio_final_value=[]
shares_history = {}
ROI = []
years = 0
annual_returns =[]
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

def generate_signals(predicted, threshold=0.01):
    signals = ['hold']
    for i in range(1, len(predicted)):
        if predicted[i] > predicted[i - 1] * (1 + threshold):
            signals.append('buy')
        elif predicted[i] < predicted[i - 1] * (1 - threshold):
            signals.append('sell')
        else:
            signals.append('hold')
    return signals

def manage_trading(stock_symbol, initial_fund):
    data = fetch_and_preprocess_stock_data(stock_symbol, "2020-01-01", "2024-02-01")
    # Feature selection
    exog_data = data[['SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'RSI', 'Bollinger_Upper', 'Bollinger_Lower']]

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
    # Predicting
    predictions = model_fit.get_forecast(steps=len(test_endog), exog=test_exog).predicted_mean


    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(train_endog.index, train_endog, label='Training Data')
    plt.plot(test_endog.index, test_endog, label='Actual Value')
    plt.plot(test_endog.index, predictions, label='Predicted Value', color='red')
    plt.title('ARIMAX Stock Price Prediction '+ stock_symbol)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Generating signals
    threshold = 0.01
    signals = generate_signals(predictions.values,  threshold)

    # Concat signals
    test_data_with_signals = test_endog.to_frame(name='Actual')
    test_data_with_signals['Predicted'] = predictions.values
    test_data_with_signals['Signal'] = signals
    portfolio_values[stock_symbol], shares_history[stock_symbol] = simulate_trading(stock_symbol, initial_fund, test_data_with_signals, signals)

    total_portfolio_value = portfolio_values[stock_symbol][-1]
    portfolio_final_value.append(total_portfolio_value)

    roi = (total_portfolio_value - initial_fund) / initial_fund
    ROI.append(roi)
    years = (test_endog.index[-1] - test_endog.index[0]).days / 255
    annual_returns.append((roi+1)/years - 1)




def simulate_trading(stock_symbol, initial_fund, test_data, signals):
    cash = initial_fund
    portfolio = {stock_symbol: {'shares': 0, 'buy_price': 0}}
    portfolio_value = [cash]
    shares_history = []

    for i in range(1, len(test_data)):
        current_price = test_data['Actual'].iloc[i]

        # Execute buy/sell actions based on signals
        if signals[i] == 'buy' and cash > 0:
            shares_to_buy = cash // current_price
            portfolio[stock_symbol]['shares'] += shares_to_buy
            portfolio[stock_symbol]['buy_price'] = current_price
            cash -= shares_to_buy * current_price
        elif signals[i] == 'sell' and portfolio[stock_symbol]['shares'] > 0:
            cash += portfolio[stock_symbol]['shares'] * current_price
            portfolio[stock_symbol]['shares'] = 0

        # Record portfolio value and shares after each trade
        portfolio_value.append(cash + portfolio[stock_symbol]['shares'] * current_price)
        shares_history.append(portfolio[stock_symbol]['shares'])

    return portfolio_value, shares_history

initial_fund =int(input('Enter Initial Investment Fund: '))
stock_symbols = list(map(str, input('Enter the list of stocks (comma seperated): ').strip().split(',')))

for stock_symbol in stock_symbols:
    manage_trading(stock_symbol, initial_fund)

mean_ar = pd.Series(annual_returns).mean()
sharpe_ratio = (mean_ar /pd.Series(annual_returns).std()) 
mean_roi= pd.Series(ROI).mean()
print(years)
# annualized_returns = (mean_roi+1)/years - 1
print(f'Total Portfolio Value: ${pd.Series(portfolio_final_value).mean()}')
print(f'ROI: {mean_roi:.2%}')
print(f'Annualized Returns: {pd.Series(annual_returns).mean():.2%}')
if len(ROI)>1:
    print(f'Sharpe Ratio: {sharpe_ratio:.4f}')

for stock_symbol in stock_symbols:
    print(f'Shares History for {stock_symbol}: {shares_history[stock_symbol]}')
