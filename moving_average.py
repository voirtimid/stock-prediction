import pandas as pd
from datetime import datetime
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

predicted_values = moving_average_algorithm(60, data)

errors = []
for i in range(len(data)):
    errors.append(data[i] - predicted_values[i])

plt.title("Moving average prediction")
plt.xlabel("Number of day")
plt.ylabel("Stock price")
plt.plot(data, label="Real Close Price")
plt.plot(predicted_values, label="Predicted Close Price")
# plt.plot(errors, label="Error")
plt.legend()
plt.savefig(f'plots/moving_average.png')
plt.show()
