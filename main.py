import pandas as pd
from datetime import datetime
from moving_average import moving_average_algorithm
import matplotlib.pyplot as plt


def date_parse(date):
    return datetime.strptime(date, '%m/%d/%Y')


df = pd.read_csv("apple_stock_prices.csv", parse_dates=True, date_parser=date_parse, index_col='Date')

df = df.sort_index(ascending=True, axis=0)

data = df["Close/Last"].tolist()

predicted_values = moving_average_algorithm(60, data)

errors = []
for i in range(len(data)):
    errors.append(data[i] - predicted_values[i])

plt.plot(data, label="Real Close Price")
plt.plot(predicted_values, label="Predicted Close Price")
plt.plot(errors, label="Error")
plt.legend()
plt.show()

