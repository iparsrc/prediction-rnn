""" Prediction application for google trends data by Parsa Akbari. """
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import date
from tensorflow import keras
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, InputLayer, Reshape, LSTM


def load_data(file_path):
    """ Loading trend data through csv files to pd.DataFrame. """
    df = pd.read_csv(file_path, sep=',', encoding='utf-8', names=['date', 'value'])
    df.set_index(keys=pd.to_datetime(df['date']), drop=True, inplace=True)
    df.drop('date', axis=1, inplace=True)
    return df

def data_preprocessing(target, window_X, window_y):
    """ Data preprocessing before fitting the model. Making squences. """
    X, y = [], []
    start_X = 0;
    end_X = start_X + window_X
    start_y = end_X
    end_y = start_y + window_y
    for _ in range(len(target)):
        if end_y < len(target):
            X.append(target[start_X:end_X])
            y.append(target[start_y:end_y])
        start_X += 1;
        end_X = start_X + window_X
        start_y += 1
        end_y = start_y + window_y
    return np.array(X), np.array(y)

def create_model(window_X, window_y):
    """ Building the model with all layers and functions. """
    model = keras.Sequential()
    # Input layer
    model.add(InputLayer(input_shape=window_X, ))

    # LSTM layer
    model.add(Reshape(target_shape=(window_X, 1)))
    model.add(LSTM(units=64, return_sequences=False))

    # Output layer
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=window_y, activation='sigmoid'))

    # Compile
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1000)
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath='model.h5', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint]

    return model, callbacks

def fit_model(model, X_train, y_train, X_test, y_test, batch_size, epochs, callbacks):
    """ Feeding the train-data to the model. """
    model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test),
              batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=1, shuffle=False)
    model.load_weights('model.h5')
    return model

def test_model(model):
    """ Testing the model with test-data and plotting results. """
    loss, acc = model.evaluate(X_test, y_test, batch_size=16, verbose=0)
    print(f"Model performence on validation data ==> loss: {loss}, accuarcy: {acc}")

def plot_data(df, pred, trend):
    """ Plotting all data. """
    idx = pd.date_range(start=date(2015, 5, 31), periods=df.shape[0], freq='W')
    df.set_index(idx, inplace=True)
    df['value'] = df['value'] * 100
    pred['LSTM'] = pred['LSTM'] * 100
    plt.plot(df['value'], label=trend)
    plt.plot(pred['LSTM'], label='LSTM Prediction')
    plt.legend()
    plt.show()

if "__main__" == __name__:
    # Define windows.
    window_X = 20
    window_y = 3

    # Load data.
    idx = 2
    csv_files = ['Facebook.csv', 'Amazon.csv', 'Google.csv', 'Oracle.csv']
    df = load_data('./data/' + csv_files[idx])

    # Scale data
    df['value'] = df['value'] / 100

    # Prepare data
    X, y = data_preprocessing(df['value'].values, window_X, window_y)

    # Training and test
    train = 180
    X_train, y_train = X[:train], y[:train]
    X_test, y_test = X[train:], y[train:]

    # Build model
    model, callbacks = create_model(window_X, window_y)

    # Fit and train model
    new_model = fit_model(model, X_train, y_train, X_test, y_test, 16, 1000, callbacks)

    # Validating model
    test_model(new_model)

    # Predecting future
    X_test = X_test[:58]
    pred = new_model.predict([X_test])
    pred_1D = list()
    for item in pred:
        pred_1D.append(item[0])

    pred_df = pd.DataFrame({'LSTM': pred_1D})
    pred_df.set_index(pd.date_range(start=date(2019, 3, 24), periods=58, freq='W'), inplace=True)

    # Plotting all historical data and precticion segment.
    plot_data(df, pred_df, csv_files[idx])