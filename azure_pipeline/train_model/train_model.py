import argparse
import os

import numpy as np

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

    # output parms
    parser.add_argument('--model', type=str, help='path to the trained model')
    parser.add_argument('--tensorboard', type=str, help='path to tensorboard files')
    
    return parser.parse_args()


def main():
    args = get_args()

    # load data
    # FIXME: the current component setting is "uri_folder" unexpectedly!
    scaler = load_scaler(os.path.join(args.scaler, 'scaler.pkl'))
    x_train = np.load(os.path.join(args.train_data_x, 'x_train.npy'))
    y_train = np.load(os.path.join(args.train_data_y, 'y_train.npy'))
    x_test = np.load(os.path.join(args.test_data_x, 'x_test.npy'))
    y_test = np.load(os.path.join(args.test_data_y, 'y_test.npy'))

    # train model
    model = build_LSTM(x_train, units=args.window)
    train_model(
        model, x_train, y_train,
        args.epochs, args.batch,
        interactive_progress=False,
        tensorboard_path=args.tensorboard
    )

    # test model
    y_lstm_scaled = model.predict(x_test)
    y_lstm = scaler.inverse_transform(y_lstm_scaled)
    y_actual = scaler.inverse_transform(y_test)

    rmse_lstm = get_rmse(pred=y_lstm, actual=y_actual)
    mape_lstm = get_mape(pred=y_lstm, actual=y_actual)
    print(f'RMSE: {rmse_lstm}, MAPE: {mape_lstm}')

    # save model
    save_model(model, args.model)


if __name__ == '__main__':
    main()
