import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb

# Load cryptocurrency data
def get_crypto_data(crypto_id, days=365):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency=usd&days={days}"
    response = requests.get(url).json()
    if 'prices' in response:
        prices = response['prices']
        data = pd.DataFrame(prices, columns=['timestamp', 'price'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        return data
    else:
        print(f"Error: 'prices' key not found in response for {crypto_id}")
        return None

# Load Bitcoin data for 365 days
btc_data = get_crypto_data('bitcoin', days=365)

# Calculate technical indicators
def moving_average(data, window):
    return data['price'].rolling(window=window).mean()

def exponential_moving_average(data, window):
    return data['price'].ewm(span=window, adjust=False).mean()

def relative_strength_index(data, window=14):
    delta = data['price'].diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['price'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['price'].ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    return macd_line, signal_line

def bollinger_bands(data, window=20):
    sma = moving_average(data, window)
    rolling_std = data['price'].rolling(window=window).std()
    upper_band = sma + (rolling_std * 2)
    lower_band = sma - (rolling_std * 2)
    return upper_band, lower_band

# Add indicators to btc_data
btc_data['SMA_20'] = moving_average(btc_data, 20)
btc_data['EMA_20'] = exponential_moving_average(btc_data, 20)
btc_data['RSI'] = relative_strength_index(btc_data, 14)
btc_data['MACD'], btc_data['Signal'] = macd(btc_data)
btc_data['Upper_Band'], btc_data['Lower_Band'] = bollinger_bands(btc_data)

# Label profitable days with a more flexible threshold
def label_profitable_days(data, threshold=0.03):  # Set lower threshold
    data['daily_return'] = data['price'].pct_change()
    data['profitable'] = (data['daily_return'] > threshold).astype(int)
    return data

btc_data = label_profitable_days(btc_data)

# Drop NaN values
btc_data = btc_data.dropna()

# Store original indices for backtesting
btc_data['original_index'] = btc_data.index

# Split data into features (X) and target (y)
X = btc_data[['SMA_20', 'EMA_20', 'RSI', 'MACD', 'Signal', 'Upper_Band', 'Lower_Band']]
y = btc_data['profitable']
original_indices = btc_data['original_index']

# Split data into training (80%) and testing (20%) sets, preserving indices
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, original_indices, test_size=0.2, shuffle=False)

# Train the LightGBM model
lgb_model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
lgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lgb_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"LightGBM Model Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Enhanced Backtesting Function
def backtest(predictions, test_indices, original_data):
    total_profit = 0
    number_of_trades = 0
    hold_days = 2  # Hold for 2 days for potential gains

    for i, pred in enumerate(predictions):
        if pred == 1:  # If predicted as profitable
            # Calculate profit over the next 'hold_days'
            for j in range(hold_days):
                if i + j < len(predictions):
                    original_index = test_indices.iloc[i + j]
                    total_profit += original_data.loc[original_index]['daily_return']
            number_of_trades += 1

    # Consider transaction costs (e.g., 0.1% per trade)
    transaction_cost = 0.001 * number_of_trades
    total_profit -= transaction_cost  # Adjust total profit for transaction costs
    return total_profit

# Perform backtest on test data
profit = backtest(y_pred, idx_test, btc_data)
print(f"Total Profit from Test Period: {profit:.2f}")