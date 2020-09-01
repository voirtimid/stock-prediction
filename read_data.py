import datetime
import pandas as pd


def date_parse(date):
    return datetime.datetime.strptime(date, '%m/%d/%Y')


def read_data():
    df = pd.read_csv("apple_stock_prices.csv", parse_dates=True, date_parser=date_parse, index_col='Date')
    data = df.sort_index(ascending=True, axis=0)

    training_set = data.iloc[:, 2:4].values

    new_data = pd.DataFrame(index=range(0, len(data)), columns=['Date', 'Close'])

    datetime.datetime.strptime(data.index[0].date().__str__(), '%Y-%m-%d')

    for i in range(0, len(data)):
        new_data['Date'][i] = i
        new_data['Close'][i] = data['Close/Last'][i]
    return new_data
