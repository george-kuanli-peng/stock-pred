import argparse
import os

import mlflow
import numpy as np

from azure_pipeline import (
    get_scaler_path,
    get_x_train_path, get_y_train_path,
    get_x_test_path, get_y_test_path
)
from train import build_LSTM, get_rmse, get_mape, load_scaler, save_model, train_model


def get_args():
    parser = argparse.ArgumentParser()

    # input parms
    parser.add_argument('--window', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--scaler', type=str, help='path to scaler')
    parser.add_argument("--train_data_x", type=str, help="path to train data x")
    parser.add_argument("--train_data_y", type=str, help="path to train data y")
    parser.add_argument("--test_data_x", type=str, help="path to test data x")
    parser.add_argument("--test_data_y", type=str, help="path to test data y")
    
    return parser.parse_args()


def main():
    args = get_args()

    # Start Logging
    mlflow.start_run()

    # load data
    # FIXME: the current component setting is "uri_folder" unexpectedly!
    scaler = load_scaler(get_scaler_path(args.scaler))
    x_train = np.load(get_x_train_path(args.train_data_x))
    y_train = np.load(get_y_train_path(args.train_data_y))
    x_test = np.load(get_x_test_path(args.test_data_x))
    y_test = np.load(get_y_test_path(args.test_data_y))

    # train model
    mlflow.tensorflow.autolog()
    model = build_LSTM(x_train, units=args.window)
    local_tensorboard_path = './tensorboard_logs'
    os.makedirs(local_tensorboard_path, exist_ok=True)
    train_model(
        model, x_train, y_train,
        args.epochs, args.batch,
        interactive_progress=False,
        tensorboard_path=local_tensorboard_path
    )

    # test model
    y_lstm_scaled = model.predict(x_test)
    y_lstm = scaler.inverse_transform(y_lstm_scaled)
    y_actual = scaler.inverse_transform(y_test)

    rmse_lstm = get_rmse(pred=y_lstm, actual=y_actual)
    mape_lstm = get_mape(pred=y_lstm, actual=y_actual)
    mlflow.log_metric('rmse', rmse_lstm)
    mlflow.log_metric('mape', mape_lstm)
    print(f'RMSE: {rmse_lstm}, MAPE: {mape_lstm}')

    # save model
    local_model_path = './saved_model'
    os.makedirs(local_model_path, exist_ok=True)
    save_model(model, local_model_path)
    mlflow.tensorflow.log_model(
        tf_saved_model_dir=local_model_path,
        tf_meta_graph_tags=None,
        tf_signature_def_key='serving_default',
        artifact_path=local_model_path,
        code_paths=['train.py']
    )

    # Stop Logging
    mlflow.end_run()


if __name__ == '__main__':
    main()
