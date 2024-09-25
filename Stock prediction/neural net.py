import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt


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


def stochastic_oscillator(data, window=14):
    low = data['price'].rolling(window=window).min()
    high = data['price'].rolling(window=window).max()
    return 100 * (data['price'] - low) / (high - low)


# Add indicators to btc_data
btc_data['SMA_20'] = moving_average(btc_data, 20)
btc_data['EMA_20'] = exponential_moving_average(btc_data, 20)
btc_data['RSI'] = relative_strength_index(btc_data, 14)
btc_data['Stochastic_Osc'] = stochastic_oscillator(btc_data, 14)


# Step 3: Dynamic Threshold for Labeling Profitable Days
def label_profitable_days(data, percentile=80):
    data['daily_return'] = data['price'].pct_change()
    threshold = np.percentile(data['daily_return'].dropna(), percentile)
    data['profitable'] = (data['daily_return'] > threshold).astype(int)
    return data


btc_data = label_profitable_days(btc_data)

# Step 4: Drop NaN values
btc_data = btc_data.dropna()

# Step 5: Store original indices for backtesting
btc_data['original_index'] = btc_data.index  # Store original indices

# Step 6: Split data into features (X) and target (y)
X = btc_data[['SMA_20', 'EMA_20', 'RSI', 'Stochastic_Osc']]  # Features
y = btc_data['profitable']  # Target
original_indices = btc_data['original_index']  # Store original indices

# Normalize features
X = (X - X.mean()) / X.std()

# Split data into training (80%) and testing (20%) sets, preserving indices
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, original_indices, test_size=0.2,
                                                                         shuffle=False)


# Step 7: Define a HyperModel for Keras Tuner
def build_model(hp):
    model = Sequential()

    # Tune the number of units in the first Dense layer
    model.add(Dense(units=hp.Int('units_1', min_value=32, max_value=512, step=32), activation='relu',
                    input_dim=X_train.shape[1]))

    # Tune the dropout rate
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))

    # Tune the number of units in the second Dense layer
    model.add(Dense(units=hp.Int('units_2', min_value=32, max_value=512, step=32), activation='relu'))

    # Tune the dropout rate
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(Dense(1, activation='sigmoid'))

    # Tune the learning rate for the optimizer
    optimizer = Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Step 8: Initialize Keras Tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,  # Try 10 different hyperparameter combinations
    executions_per_trial=1,  # Run each trial once
    directory='my_dir',
    project_name='crypto_nn_tuning'
)

# Step 9: Perform the hyperparameter search
tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Step 10: Retrieve the best hyperparameters and model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
Best Hyperparameters:
- Units (Layer 1): {best_hps.get('units_1')}
- Dropout (Layer 1): {best_hps.get('dropout_1')}
- Units (Layer 2): {best_hps.get('units_2')}
- Dropout (Layer 2): {best_hps.get('dropout_2')}
- Learning Rate: {best_hps.get('learning_rate')}
""")

# Step 11: Build and train the model with the best hyperparameters
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Step 12: Make predictions on the test set
y_pred = (best_model.predict(X_test) > 0.5).astype(int).flatten()

# Step 13: Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Neural Network Model Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Step 14: Backtesting Function (using neural network)
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

