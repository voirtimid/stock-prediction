import pandas as pd
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import tensorflow
import matplotlib.pyplot as plt


def date_parse(date):
    return datetime.datetime.strptime(date, '%m/%d/%Y')


df = pd.read_csv("apple_stock_prices.csv", parse_dates=True, date_parser=date_parse, index_col='Date')

data = df.sort_index(ascending=True, axis=0)

training_set = data.iloc[:, 2:4].values

sc = MinMaxScaler(feature_range=(0, 1))
training_data_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(60, 2035):
    X_train.append(training_data_scaled[i - 60:i, 0])
    y_train.append(training_data_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

regressor = Sequential()

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# steps from link ->
# https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/?

new_data = pd.DataFrame(index=range(0, len(data)), columns=['Date', 'Close'])

datetime.datetime.strptime(data.index[0].date().__str__(), '%Y-%m-%d')

for i in range(0, len(data)):
    new_data['Date'][i] = data.index[i]
    new_data['Close'][i] = data['Close/Last'][i]
