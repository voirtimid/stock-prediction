import math

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from read_data import read_data

data = read_data()

# setting index
data.index = data.Date
data.drop('Date', axis=1, inplace=True)

# creating train and test sets
dataset = data.values

data_size = len(data)
split_factor = math.floor(data_size * 0.8)

train = dataset[0:split_factor, :]
valid = dataset[split_factor:, :]

# converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i - 60:i, 0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=2)

inputs = data[len(data) - len(valid) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

mse = mean_squared_error(valid, closing_price)
rmse = np.sqrt(mse)
print(rmse)

train = data[:split_factor]
valid = data[split_factor:]

valid['Prediction'] = closing_price

# plot
plt.title("LSTM")
plt.xlabel("Number of day")
plt.ylabel("Stock price")
plt.plot(train['Close'], label='Train Close price')
plt.plot(valid['Close'], label='Real Close price')
plt.plot(valid['Prediction'], label='Prediction Close price')
plt.legend()
plt.savefig(f'plots/lstm.png')
plt.show()
