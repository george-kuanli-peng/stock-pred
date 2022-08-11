import argparse
import csv
import json
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model as tf_load_model


_track = None


class MLSteamModelTrainingTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        track = get_mlsteam_track()
        track['metrics/train'].log(json.dumps({'epoch': epoch, 'loss': logs['loss']}))
        track['metrics/train.chart'].log(json.dumps({'epoch': epoch, 'loss': logs['loss']}))


def get_mlsteam_track():
    global _track
    if _track is None:
        import mlsteam
        _track = mlsteam.init()
    return _track


def load_data(pkl_file_path):
    with open(pkl_file_path, 'rb') as fin:
        data = pickle.load(fin)

    stock_dates = np.array([h['date'] for h in data['prices']],
                           dtype='datetime64[s]')  # in timestamp
    stock_dates = np.flipud(stock_dates)
    stock_prices = np.array([h['close'] for h in data['prices']])
    stock_prices = np.flipud(stock_prices)

    return stock_dates, stock_prices


def load_data_csv(csv_file_path):
    stock_dates, stock_prices = [], []
    with open(csv_file_path, 'rt', newline='') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            stock_dates.append(int(row['timestamp']))
            stock_prices.append(float(row['close']))

    stock_dates = np.array(stock_dates, dtype='datetime64[s]')
    stock_dates = np.flipud(stock_dates)
    stock_prices = np.array(stock_prices)
    stock_prices = np.flipud(stock_prices)

    return stock_dates, stock_prices


def transform_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[:, None])
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
                interactive_progress=True, tensorboard_path=None,
                mlsteam_track=False):
    if mlsteam_track:
        track = get_mlsteam_track()
        track['params/epochs'] = epochs
        track['params/batch'] = batch
        track['data/x_train'] = x_train
        track['data/y_train'] = y_train

    callbacks = []
    if tensorboard_path:
        cb = TensorBoard(log_dir=tensorboard_path, histogram_freq=1)
        callbacks.append(cb)
    if mlsteam_track:
        callbacks.append(MLSteamModelTrainingTracker())
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlsteam_track', action='store_true')
    return parser.parse_args()


def main():
    from mlsteam import stparams
    # Global settings
    WINDOW = int(stparams.get_value('window', 50))
    EPOCHS = int(stparams.get_value('epochs', 15))
    BATCH = int(stparams.get_value('batch', 20))
    TEST_RATIO = float(stparams.get_value('test_ratio', .2))

    DATA_PATH = stparams.get_value(
        'data_path', '/mlsteam/data/stock_prices/20220512_tesla.pkl')
    SCALER_PATH = stparams.get_value('scaler_path', '/working/scaler.pkl')
    MODEL_PATH = stparams.get_value('model_path', '/working/1')
    TENSORBOARD_PATH = stparams.get_value('tensorboard_path', '/working/tensorboard')

    args = get_args()

    # Forces to use CPU rather than GPU
    # NVIDIA drivers of higher versions have messy implimentation of LSTM!
    # Ref: https://github.com/mozilla/DeepSpeech/issues/3088#issuecomment-656056969
    # Ref: https://github.com/tensorflow/tensorflow/issues/35950#issuecomment-577427083
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    stock_dates, stock_prices = load_data(DATA_PATH)
    scaler, scaled_data = transform_data(stock_prices)

    train_split = int(len(scaled_data) * (1.0 - TEST_RATIO))
    scaled_data_train = scaled_data[:train_split]
    x_train, y_train = extract_x_y(
        scaled_data_train, window=WINDOW, offset=WINDOW)
    model = build_LSTM(x_train, units=WINDOW)
    train_model(model, x_train, y_train, EPOCHS, BATCH,
                interactive_progress=True, tensorboard_path=TENSORBOARD_PATH,
                mlsteam_track=args.mlsteam_track)
    save_scaler(scaler, SCALER_PATH)
    save_model(model, MODEL_PATH)


if __name__ == '__main__':
    main()
