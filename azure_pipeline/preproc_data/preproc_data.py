import argparse
import glob
import os

import numpy as np

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

    # read data
    # FIXME: the current component setting is "uri_folder" unexpectedly!
    data_file = glob.glob(os.path.join(args.data, '*.pkl'))[0]  # get the first matched file
    stock_dates, stock_prices = load_data(data_file)
    scaler, scaled_data = transform_data(stock_prices)

    # split and preproc data
    train_split = int(len(scaled_data) * (1.0 - args.test_ratio))
    scaled_data_train = scaled_data[:train_split]
    x_train, y_train = extract_x_y(scaled_data_train, window=args.window, offset=args.window)
    x_test, y_test = extract_x_y(scaled_data, window=args.window, offset=train_split)

    # save data
    # FIXME: the current component setting is "uri_folder" unexpectedly!
    save_scaler(scaler, os.path.join(args.scaler, 'scaler.pkl'))
    np.save(os.path.join(args.train_data_x, 'x_train.npy'), x_train)
    np.save(os.path.join(args.train_data_y, 'y_train.npy'), y_train)
    np.save(os.path.join(args.test_data_x, 'x_test.npy'), x_test)
    np.save(os.path.join(args.test_data_y, 'y_test.npy'), y_test)

if __name__ == '__main__':
    main()
