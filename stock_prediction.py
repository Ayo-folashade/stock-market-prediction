import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Read the CSV file containing Netflix stock price data
data = pd.read_csv('NFLX.csv')

# Extract the 'Close' price column as the target variable
target = data['Close'].values.reshape(-1, 1)

# Convert the target variable to float32
target = target.astype('float32')

# Scale the target variable using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
target = scaler.fit_transform(target)

# Define the length of input sequences
sequence_length = 10  # Adjust as per your requirements

# Split the data into training and testing sets
train_data = target[:800]
test_data = target[800:]


# Function to create input sequences and corresponding target values
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y).reshape(-1, 1)  # Reshape y to (n_samples, 1)


# Create input sequences and target values for training and testing data
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Reshape the input data to match the LSTM input shape
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(sequence_length, 1)))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64)

# Evaluate the model on the test data
loss = model.evaluate(X_test, y_test)
print("Test loss:", loss)

# Make predictions on the test data
predictions = model.predict(X_test)

# Inverse scale the predictions
predictions = scaler.inverse_transform(predictions)

# Print the predicted and actual prices
for i in range(len(predictions)):
    predicted_price = predictions[i][0]
    actual_price = scaler.inverse_transform(np.reshape(y_test[i], (1, -1)))[0][0]
    print("Predicted price:", predicted_price, "Actual price:", actual_price)
