import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.models import load_model

def predict_gru(data,isROC):
    df = pd.read_csv(data)
    df.head()
    file_name = data.split(sep='.')
    df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df.index = df['Date']
    num_shape = 2000
    if not isROC:
        data = df.sort_index(ascending=True, axis=0)
        new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

        for i in range(0, len(data)):
            new_dataset['Date'][i] = data['Date'][i]
            new_dataset['Close'][i] = data['Close'][i]

        new_dataset.index = new_dataset.Date
        new_dataset.drop('Date', axis=1, inplace=True)

        final_dataset = new_dataset.values

        train_data = final_dataset[:num_shape, :]
        valid_data = final_dataset[num_shape:, :]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(final_dataset)

        x_train_data, y_train_data = [], []

        for i in range(60, len(train_data)):
            x_train_data.append(scaled_data[i - 60:i, 0])
            y_train_data.append(scaled_data[i, 0])

        x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

        x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))
        gru_model = Sequential()
        if os.path.isfile('gru_close' + file_name[0] + '.h5'):
            gru_model = load_model('gru_close' + file_name[0] + '.h5')
        else:
            gru_model.add(GRU(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
            gru_model.add(GRU(units=50))
            gru_model.add(Dense(1))

            gru_model.compile(loss='mean_squared_error', optimizer='adam')
            gru_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)
            gru_model.save('gru_close' + file_name[0] + '.h5')

        inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
        inputs_data = inputs_data.reshape(-1, 1)
        inputs_data = scaler.transform(inputs_data)

        X_test = []
        for i in range(60, inputs_data.shape[0]):
            X_test.append(inputs_data[i - 60:i, 0])
        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        closing_price = gru_model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)

        train_data = new_dataset[:num_shape]
        valid_data = new_dataset[num_shape:]
        valid_data['Predictions'] = closing_price

        return df,valid_data

    else:
        data = df.sort_index(ascending=True, axis=0)
        df['Close_Yesterday'] = df['Close'].shift(1)
        df['ROC'] = (df['Close'] - df['Close_Yesterday'])* 100 / df['Close_Yesterday']
        df['Target'] = df['ROC']
        new_dataset = pd.DataFrame(index=range(1, len(df)), columns=['Date', 'Close'])

        for i in range(0, len(data)):
            new_dataset["Date"][i] = data['Date'][i]
            new_dataset["Close"][i] = data["Close"][i]

        new_dataset.index = new_dataset.Date
        new_dataset.drop("Date", axis=1, inplace=True)

        final_dataset = new_dataset.values

        train_data = final_dataset[0:num_shape, :]
        valid_data = final_dataset[num_shape:, :]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(final_dataset)

        x_train_data, y_train_data = [], []

        for i in range(60, len(train_data)):
            x_train_data.append(scaled_data[i - 60:i, 0])
            y_train_data.append(scaled_data[i, 0])

        x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

        x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))
        if os.path.isfile('gru_roc' + file_name[0] + '.h5'):
            gru_model = load_model('gru_roc' + file_name[0] + '.h5')
        else:
            gru_model = Sequential()
            gru_model.add(GRU(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
            gru_model.add(GRU(units=50))
            gru_model.add(Dense(1))

            gru_model.compile(loss='mean_squared_error', optimizer='adam')
            gru_model.fit(x_train_data, y_train_data, epochs=5, batch_size=1, verbose=2)
            gru_model.save('gru_roc' + file_name[0] + '.h5')

        inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
        inputs_data = inputs_data.reshape(-1, 1)
        inputs_data = scaler.transform(inputs_data)

        X_test = []
        for i in range(60, inputs_data.shape[0]):
            X_test.append(inputs_data[i - 60:i, 0])
        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        closing_price = gru_model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)

        train_data = new_dataset[:num_shape]
        valid_data = new_dataset[num_shape:]
        valid_data['Predictions'] = closing_price

        new_dataset["Close_Yesterday"] = new_dataset["Close"].shift(1)
        new_dataset['ROC'] = (new_dataset['Close'] - new_dataset['Close_Yesterday']) * 100 / new_dataset[
            'Close_Yesterday']

        valid_data['Yesterday_Predictions'] = valid_data['Predictions'].shift(1)
        valid_data['Yesterday_Close'] = valid_data['Close'].shift(1)
        valid_data['Predict_ROC'] = (valid_data['Predictions'] - valid_data['Yesterday_Predictions']) * 100 / \
                                    valid_data['Yesterday_Predictions']
        valid_data['Close_ROC'] = (valid_data['Close'] - valid_data['Yesterday_Close']) * 100 / valid_data[
            'Yesterday_Close']

        return df,valid_data

def main():
    df, val = predict_gru('TSLA.csv',False)
    print(df)
    print(val)

if __name__ == '__main__':
    main()
