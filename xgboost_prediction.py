import numpy as np
import pandas as pd
import datetime
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

df = pd.read_csv('./FB.csv') # MUST HAVE "Date" & "Close" column (with capital first letter)

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
