import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model as tf_load_model


def load_data(pkl_file_path):
    with open(pkl_file_path, 'rb') as fin:
        data = pickle.load(fin)
    
    stock_dates = np.array([h['date'] for h in data['prices']], dtype='datetime64[s]')  # in timestamp
    stock_dates = np.flipud(stock_dates)
    stock_prices = np.array([h['close'] for h in data['prices']])
    stock_prices = np.flipud(stock_prices)
    
    return stock_dates, stock_prices


def transform_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[:,None])
    return scaler, scaled_data


def extract_x_y(data, window, offset):
    # Ref: https://neptune.ai/blog/predicting-stock-prices-using-machine-learning
    x, y = [], []
    for i in range(offset, len(data)):
        x.append(data[i-window:i])
        y.append(data[i])
    return np.array(x), np.array(y)


def build_LSTM(x_train, units):
    l_in = Input(shape=(x_train.shape[1], 1))
    l_hid = LSTM(units=units, return_sequences=True)(l_in)
    l_hid = LSTM(units=units)(l_hid)
    l_out = Dense(units=1, activation='linear')(l_hid)
    model = Model(l_in, l_out)
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def train_model(model, x_train, y_train, epochs, batch,
                interactive_progress=True, tensorboard_path=None):
    callbacks = []
    if tensorboard_path:
        cb = TensorBoard(log_dir=tensorboard_path, histogram_freq=1)
        callbacks.append(cb)
    history = model.fit(x_train, y_train,
                        epochs=epochs, batch_size=batch,
                        verbose=1 if interactive_progress else 2,
                        validation_split=.1, shuffle=True,
                        callbacks=callbacks)
    return history


def predict(model, x_test):
    return model.predict(x_test)


def save_scaler(scaler, pkl_file_path):
    path = Path(pkl_file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_file_path, 'wb') as fout:
        pickle.dump(scaler, fout)


def load_scaler(pkl_file_path):
    scaler = None
    with open(pkl_file_path, 'rb') as fin:
        scaler = pickle.load(fin)
    return scaler


def save_model(model, model_dir):
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.save(model_dir)


def load_model(model_dir):
    return tf_load_model(model_dir)


def get_rmse(pred, actual):
    return np.sqrt(np.mean((pred-actual)**2))


def get_mape(pred, actual):
    return (np.fabs(actual-pred)/actual)[actual != 0].mean()
