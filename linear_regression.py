import numpy as np
import math
import matplotlib.pyplot as plt
from read_data import read_data
from sklearn.linear_model import LinearRegression

data = read_data()

data_size = len(data)
split_factor = math.floor(data_size * 0.8)

train = data[:split_factor]
valid = data[split_factor + 1:]

x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']

model = LinearRegression()
model.fit(x_train, y_train)

# make predictions and find the rmse
preds = model.predict(x_valid)
rms = np.sqrt(np.mean(np.power((np.array(y_valid) - np.array(preds)), 2)))
print(rms)

# plot
valid['Predictions'] = 0
valid['Predictions'] = preds

valid.index = data[split_factor + 1:].index
train.index = data[:split_factor].index

plt.title("Linear regression prediction")
plt.xlabel("Number of day")
plt.ylabel("Stock price")
plt.plot(train['Close'], label='Train Close price')
plt.plot(valid['Close'], label='Real Close price')
plt.plot(valid['Predictions'], label='Prediction Close price')
plt.legend()
plt.savefig(f'plots/linear_regression.png')
plt.show()
