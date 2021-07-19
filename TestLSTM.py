# Đọc kỹ yêu cầu trước khi chạy script
# Thư viện tensorflow bị lỗi với phiên bản numpy mới nhất (1.20.x)
# Cần cài dặt lại numpy với bản 1.18.5 theo hướng dẫn sau:
# https://stackoverflow.com/questions/66207609/notimplementederror-cannot-convert-a-symbolic-tensor-lstm-2-strided-slice0-t





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from matplotlib.pylab import rcParams
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler

rcParams['figure.figsize'] = 20, 10


def predict_lstm(data, isROC):
    df = pd.read_csv(data)
    df.head()

    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df.index = df['Date']

    # plt.figure(figsize=(16, 8))
    # plt.plot(df["Close"], label='Close Price history')
    # plt.legend()
    if not isROC:
        data = df.sort_index(ascending=True, axis=0)
        new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

        for i in range(0, len(data)):
            new_dataset["Date"][i] = data['Date'][i]
            new_dataset["Close"][i] = data["Close"][i]

        new_dataset.index = new_dataset.Date
        new_dataset.drop("Date", axis=1, inplace=True)

        final_dataset = new_dataset.values

        train_data = final_dataset[0:987, :]
        valid_data = final_dataset[987:, :]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(final_dataset)

        x_train_data, y_train_data = [], []

        for i in range(60, len(train_data)):
            x_train_data.append(scaled_data[i - 60:i, 0])
            y_train_data.append(scaled_data[i, 0])

        x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

        x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

        lstm_model = Sequential()
        lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
        lstm_model.add(LSTM(units=50))
        lstm_model.add(Dense(1))

        lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)

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

        lstm_model.save("saved_lstm_model.h5")

        train_data = new_dataset[:987]
        valid_data = new_dataset[987:]
        valid_data['Predictions'] = closing_price
        plt.plot(train_data["Close"], label="Train_data_close")
        plt.plot(valid_data['Close'], label="Valid_data_close")
        plt.plot(valid_data['Predictions'], label="valid_data_predictions")
        plt.legend()

    else:
        df = df[['Close', 'Date']].copy()
        df['Close_Yesterday'] = df['Close'].shift(1)
        df['ROC'] = (df['Close'] - df['Close_Yesterday']) / df['Close_Yesterday']
        df['Target'] = df['ROC'].shift(-1)
        df = df[['ROC', 'Date', 'Target']].copy()

        data = df.sort_index(ascending=True, axis=0)
        new_dataset = pd.DataFrame(index=range(1, len(df)), columns=['Date', 'Target'])

        for i in range(0, len(data)):
            new_dataset["Date"][i] = data['Date'][i]
            # new_dataset["Close"][i] = data["Close"][i]
            new_dataset["Target"][i] = data["Target"][i]

        new_dataset.index = new_dataset.Date
        new_dataset.drop("Date", axis=1, inplace=True)

        final_dataset = new_dataset.values

        train_data = final_dataset[0:987, :]
        valid_data = final_dataset[987:, :]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(final_dataset)

        x_train_data, y_train_data = [], []

        for i in range(60, len(train_data)):
            x_train_data.append(scaled_data[i - 60:i, 0])
            y_train_data.append(scaled_data[i, 0])

        x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

        x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

        lstm_model = Sequential()
        lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
        lstm_model.add(LSTM(units=50))
        lstm_model.add(Dense(1))

        lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)

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

        lstm_model.save("saved_lstm_model.h5")

        train_data = new_dataset[:987]
        valid_data = new_dataset[987:]
        valid_data['Predictions'] = closing_price
        plt.plot(train_data["Target"], label="Train_data_close")
        plt.plot(valid_data['Target'], label="Valid_data_close")
        plt.plot(valid_data['Predictions'], label="valid_data_predictions")
        plt.legend()


predict_lstm("C:/Users/DELL-PC/Documents/FB.csv", True)
