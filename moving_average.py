import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def date_parse(date):
    return datetime.strptime(date, '%m/%d/%Y')


def moving_average_algorithm(factor: int, data: []):
    predicted_values_result = []
    for i in range(len(data)):
        if i <= factor:
            predicted_values_result.append(data[i])
        else:
            average_value: float = sum(data[i-factor:i]) / factor
            predicted_values_result.append(average_value)

    return predicted_values_result


df = pd.read_csv("apple_stock_prices.csv", parse_dates=True, date_parser=date_parse, index_col='Date')

df = df.sort_index(ascending=True, axis=0)

data = df["Close/Last"].tolist()

predicted_values = moving_average_algorithm(10, data)

errors = []
for i in range(len(data)):
    errors.append(data[i] - predicted_values[i])

mse = mean_squared_error(data, errors)
rmse = np.sqrt(mse)

print(mse)
print(rmse)

plt.title("Moving average (10)")
plt.xlabel("Number of day")
plt.ylabel("Stock price")
plt.plot(data, label="Real Close Price")
plt.plot(predicted_values, label="Predicted Close Price")
# plt.plot(errors, label="Error")
plt.legend()
plt.savefig(f'plots/moving_average_10_error.png')
plt.show()
