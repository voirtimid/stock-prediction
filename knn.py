import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from read_data import read_data
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

data = read_data()

scaler = MinMaxScaler(feature_range=(0, 1))

data_size = len(data)
split_factor = math.floor(data_size * 0.8)

train = data[:split_factor]
valid = data[split_factor:]

x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']

# scaling data
x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_valid_scaled = scaler.fit_transform(x_valid)
x_valid = pd.DataFrame(x_valid_scaled)

# using gridsearch to find the best parameter
params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

# fit the model and make predictions
model.fit(x_train, y_train)
preds = model.predict(x_valid)

rms = np.sqrt(np.mean(np.power((np.array(y_valid) - np.array(preds)), 2)))
print(rms)

valid['Predictions'] = 0
valid['Predictions'] = preds

# plot
plt.title("K-nearest neighbours")
plt.xlabel("Number of day")
plt.ylabel("Stock price")
plt.plot(train['Close'], label='Train Close price')
plt.plot(valid['Close'], label='Real Close price')
plt.plot(valid['Predictions'], label='Prediction Close price')
plt.legend()
plt.savefig(f'plots/knn.png')
plt.show()
