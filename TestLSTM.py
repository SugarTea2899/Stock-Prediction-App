# Đọc kỹ yêu cầu trước khi chạy script
# Thư viện tensorflow bị lỗi với phiên bản numpy mới nhất (1.20.x)
# Cần cài dặt lại numpy với bản 1.18.5 theo hướng dẫn sau:
# https://stackoverflow.com/questions/66207609/notimplementederror-cannot-convert-a-symbolic-tensor-lstm-2-strided-slice0-t

import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from matplotlib.pylab import rcParams
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
pd.options.mode.chained_assignment = None

rcParams['figure.figsize'] = 20, 10


def predict_lstm(data, isROC):
    df = pd.read_csv(data)
    df.head()
    file_name = data.split(sep='.')
    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df.index = df['Date']


    if not isROC:
        data = df.sort_index(ascending=True, axis=0)
        new_dataset = pd.DataFrame(index=range(1, len(df)), columns=['Date', 'Close'])

        for i in range(0, len(data)):
            new_dataset["Date"][i] = data['Date'][i]
            new_dataset["Close"][i] = data["Close"][i]

        new_dataset.index = new_dataset.Date
        new_dataset.drop("Date", axis=1, inplace=True)

        final_dataset = new_dataset.values

        dataset_length = math.floor(len(final_dataset) * 0.7)

        train_data = final_dataset[0:dataset_length, :]
        valid_data = final_dataset[dataset_length:, :]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(final_dataset)

        x_train_data, y_train_data = [], []

        for i in range(60, len(train_data)):
            x_train_data.append(scaled_data[i - 60:i, 0])
            y_train_data.append(scaled_data[i, 0])

        x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

        x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))
        if os.path.isfile('lstm_close' + file_name[0] + '.h5'):
            lstm_model = load_model('lstm_close' + file_name[0] + '.h5')
        else:
            lstm_model = Sequential()
            lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
            lstm_model.add(LSTM(units=50))
            lstm_model.add(Dense(1))
    
            lstm_model.compile(loss='mean_squared_error', optimizer='adam')
            lstm_model.fit(x_train_data, y_train_data, epochs=5, batch_size=1, verbose=2)
            lstm_model.save('lstm_close' + file_name[0] + '.h5')

        inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
        inputs_data = inputs_data.reshape(-1, 1)
        inputs_data = scaler.transform(inputs_data)

        X_test = []
        for i in range(60, inputs_data.shape[0]):
            X_test.append(inputs_data[i - 60:i, 0])
        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        closing_price = lstm_model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)

        train_data = new_dataset[:dataset_length]
        valid_data = new_dataset[dataset_length:]
        valid_data['Predictions'] = closing_price

        return df, valid_data;



    else:
        data = df.sort_index(ascending=True, axis=0)
        df['Close_Yesterday'] = df['Close'].shift(1)
        df['ROC'] = (df['Close'] - df['Close_Yesterday']) *100/ df['Close_Yesterday']
        df['Target'] = df['ROC']
        new_dataset = pd.DataFrame(index=range(1, len(df)), columns=['Date', 'Close'])

        for i in range(0, len(data)):
            new_dataset["Date"][i] = data['Date'][i]
            new_dataset["Close"][i] = data["Close"][i]

        new_dataset.index = new_dataset.Date
        new_dataset.drop("Date", axis=1, inplace=True)

        final_dataset = new_dataset.values
        dataset_length = math.floor(len(final_dataset) * 0.7)

        train_data = final_dataset[0:dataset_length, :]
        valid_data = final_dataset[dataset_length:, :]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(final_dataset)

        x_train_data, y_train_data = [], []

        for i in range(60, len(train_data)):
            x_train_data.append(scaled_data[i - 60:i, 0])
            y_train_data.append(scaled_data[i, 0])

        x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

        x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))
        if os.path.isfile('lstm_roc' + file_name[0] + '.h5'):
            lstm_model = load_model('lstm_roc' + file_name[0] + '.h5')
        else:
            lstm_model = Sequential()
            lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
            lstm_model.add(LSTM(units=50))
            lstm_model.add(Dense(1))
    
            lstm_model.compile(loss='mean_squared_error', optimizer='adam')
            lstm_model.fit(x_train_data, y_train_data, epochs=10, batch_size=1, verbose=2)
            lstm_model.save('lstm_roc' + file_name[0] + '.h5')

        inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
        inputs_data = inputs_data.reshape(-1, 1)
        inputs_data = scaler.transform(inputs_data)

        X_test = []
        for i in range(60, inputs_data.shape[0]):
            X_test.append(inputs_data[i - 60:i, 0])
        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        closing_price = lstm_model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)

        train_data = new_dataset[:dataset_length]
        valid_data = new_dataset[dataset_length:]
        valid_data['Predictions'] = closing_price

        print("new_dataset")
        print(new_dataset)

        new_dataset["Close_Yesterday"] = new_dataset["Close"].shift(1)
        new_dataset['ROC'] = (new_dataset['Close'] - new_dataset['Close_Yesterday']) * 100 / new_dataset[
            'Close_Yesterday']

        valid_data['Yesterday_Predictions'] = valid_data['Predictions'].shift(1)
        valid_data['Yesterday_Close'] = valid_data['Close'].shift(1)
        valid_data['Predict_ROC'] = (valid_data['Predictions'] - valid_data['Yesterday_Predictions']) * 100 / \
                                    valid_data['Yesterday_Predictions']
        valid_data['Close_ROC'] = (valid_data['Close'] - valid_data['Yesterday_Close']) * 100 / valid_data[
            'Yesterday_Close']

        return df, valid_data;

