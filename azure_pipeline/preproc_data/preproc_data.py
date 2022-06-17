import argparse

import mlflow
import numpy as np

from azure_pipeline import (
    get_first_match_data_file,
    get_scaler_path,
    get_x_train_path, get_y_train_path,
    get_x_test_path, get_y_test_path
)
from train import extract_x_y, load_data, save_scaler, transform_data


def get_args():
    parser = argparse.ArgumentParser()

    # input parms
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_ratio", type=float, required=False, default=0.2)
    parser.add_argument('--window', type=int)

    # output parms
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

    # read data
    # FIXME: the current component setting is "uri_folder" unexpectedly!
    data_file = get_first_match_data_file(args.data)  # get the first matched file
    stock_dates, stock_prices = load_data(data_file)
    scaler, scaled_data = transform_data(stock_prices)

    # split and preproc data
    train_split = int(len(scaled_data) * (1.0 - args.test_ratio))
    scaled_data_train = scaled_data[:train_split]
    x_train, y_train = extract_x_y(scaled_data_train, window=args.window, offset=args.window)
    x_test, y_test = extract_x_y(scaled_data, window=args.window, offset=train_split)

    mlflow.log_metric('num_samples', scaled_data.shape[0])
    mlflow.log_metric('test_ratio', args.test_ratio)

    # save data
    # FIXME: the current component setting is "uri_folder" unexpectedly!
    save_scaler(scaler, get_scaler_path(args.scaler))
    np.save(get_x_train_path(args.train_data_x), x_train)
    np.save(get_y_train_path(args.train_data_y), y_train)
    np.save(get_x_test_path(args.test_data_x), x_test)
    np.save(get_y_test_path(args.test_data_y), y_test)

    # Stop Logging
    mlflow.end_run()

if __name__ == '__main__':
    main()
