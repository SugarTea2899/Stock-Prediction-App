import numpy as np
import pandas as pd
import datetime
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

def train_test_split(data, perc):
    data = data.values
    n = int(len(data) * (1 - perc))
    return data[:n], data[n:]

def xgboost_predict(train, val):
    train = np.array(train)
    x = train[:, :-2]
    y = train[:, -1]

    model = XGBRegressor(objective="reg:squarederror", n_estimators=1000)
    model.fit(x, y)

    val = np.array(val).reshape(1, -1)
    pred = model.predict(val)
    return pred[0]

def xgboost_prediction(df, isROC):

    if (isROC):
        df = df[['Close', 'Date']].copy()
        df['Close_Yesterday'] = df.Close.shift(1)
        df['ROC'] = (df['Close'] - df['Close_Yesterday']) / \
            df['Close_Yesterday']
        df['Target'] = df.ROC.shift(-1)
        df = df[['ROC', 'Date', 'Target']].copy()
    else:
        df = df[['Close', 'Date']].copy()
        df['Target'] = df.Close.shift(-1)

    df.dropna(inplace=True)
    prediction = []
    precentTestRecord = 0.015  # update this
    train, test = train_test_split(df, precentTestRecord)

    history = [x for x in train]

    for i in range(len(test)):
        val = np.array(test[i, :-2])

        pred = xgboost_predict(history, val)
        prediction.append(pred)

        history.append(test[i])
        print('%d/%d test done' % (i + 1, len(test)))

    predDate = test[:, 1][1:]
    predDate = [x for x in predDate]

    lastDateStr = predDate[len(predDate) - 1]
    lastDate = datetime.datetime.strptime(lastDateStr, '%Y-%m-%d')

    nextDate = lastDate + datetime.timedelta(days=1)
    nextDateStr = datetime.datetime.strftime(nextDate, '%Y-%m-%d')
    predDate.append(nextDateStr)

    prediction_data = {'Date': predDate, 'Prediction': prediction}
    prediction_df = pd.DataFrame(prediction_data)

    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df.index = df['Date']

    prediction_df["Date"] = pd.to_datetime(
        prediction_df.Date, format="%Y-%m-%d")
    prediction_df.index = prediction_df['Date']
    return df, prediction_df
