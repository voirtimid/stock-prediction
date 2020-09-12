import math
import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
import matplotlib.pyplot as plt
from read_data import read_data
from sklearn.metrics import mean_squared_error

data = read_data()

data_size = len(data)
split_factor = math.floor(data_size * 0.8)

train = data[:split_factor]
valid = data[split_factor:]

training = train['Close']
validation = valid['Close']

model = auto_arima(training, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(training)

predictions = model.predict(n_periods=504)
predictions = pd.DataFrame(predictions, index=valid.index, columns=['Prediction'])

mse = mean_squared_error(np.array(valid['Close']), np.array(predictions['Prediction']))
rmse = np.sqrt(mse)
print(rmse)


# plot
plt.title("Auto ARIMA")
plt.xlabel("Number of day")
plt.ylabel("Stock price")
plt.plot(train['Close'], label='Train Close price')
plt.plot(valid['Close'], label='Real Close price')
plt.plot(predictions['Prediction'], label='Prediction Close price')
plt.legend()
plt.savefig(f'plots/auto_arima.png')
plt.show()
