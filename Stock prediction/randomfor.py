import requests
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load cryptocurrency data
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
if btc_data is None:
    raise ValueError("Failed to load cryptocurrency data.")

# Step 2: Calculate technical indicators
def moving_average(data, window):
    return data['price'].rolling(window=window).mean()

def relative_strength_index(data, window=14):
    delta = data['price'].diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Add indicators to btc_data
btc_data['SMA_20'] = moving_average(btc_data, 20)
btc_data['RSI'] = relative_strength_index(btc_data, 14)

# Step 3: Label profitable days
def label_profitable_days(data, threshold=0.04):
    data['daily_return'] = data['price'].pct_change()
    data['profitable'] = (data['daily_return'] > threshold).astype(int)
    return data

btc_data = label_profitable_days(btc_data)

# Step 4: Drop NaN values
btc_data = btc_data.dropna()

# Step 5: Store original indices for backtesting
btc_data['original_index'] = btc_data.index  # Store original indices

# Step 6: Split data into features (X) and target (y)
X = btc_data[['SMA_20', 'RSI']]  # Features
y = btc_data['profitable']        # Target
original_indices = btc_data['original_index']  # Store original indices

# Split data into training (80%) and testing (20%) sets, preserving indices
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, original_indices, test_size=0.2, shuffle=False)

# Step 7: Train the RandomForest model
model = RandomForestClassifier(n_estimators=10000, random_state=42)
model.fit(X_train, y_train)

# Step 8: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 9: Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 10: Backtesting Function
def backtest(predictions, test_indices, original_data):
    profit = 0
    for i, pred in enumerate(predictions):
        if pred == 1:  # If predicted as profitable
            original_index = test_indices.iloc[i]  # Map back to original btc_data index
            profit += original_data.loc[original_index]['daily_return']  # Add actual return
    return profit

# Perform backtest on test data
profit = backtest(y_pred, idx_test, btc_data)
print(f"Total Profit from Test Period: {profit:.2f}")
