import glob
import os


def get_first_match_data_file(base_dir):
    return glob.glob(os.path.join(base_dir, '*.pkl'))[0]


def get_scaler_path(base_dir):
    return os.path.join(base_dir, 'scaler.pkl')


def get_x_train_path(base_dir):
    return os.path.join(base_dir, 'x_train.npy')


def get_y_train_path(base_dir):
    return os.path.join(base_dir, 'y_train.npy')


def get_x_test_path(base_dir):
    return os.path.join(base_dir, 'x_test.npy')


def get_y_test_path(base_dir):
    return os.path.join(base_dir, 'y_test.npy')
