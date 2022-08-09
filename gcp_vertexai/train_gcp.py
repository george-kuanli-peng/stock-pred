import argparse
import logging
import os
from datetime import datetime

from google.cloud import aiplatform

from train import (build_LSTM, extract_x_y, load_data_csv, train_model, transform_data)


logger = logging.getLogger('logger')
logging.basicConfig(level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--project_id', type=str)
    parser.add_argument('--location', type=str)
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--window', '-W', type=int, default=50)
    parser.add_argument('--epoch', '-E', type=int, default=15)
    parser.add_argument('--batch', '-B', type=int, default=20)
    parser.add_argument('--test_ratio', '-R', type=float, default=.2)
    return parser.parse_args()


def main(experiment_name: str, project_id: str, location: str,
         train_csv: str,
         window: int, epoch: int, batch: int, test_ratio: float):
    aiplatform.init(project=project_id, location=location)
    # tb = aiplatform.Tensorboard.create()
    # aiplatform.init(experiment=experiment_name, experiment_tensorboard=tb)
    aiplatform.init(experiment=experiment_name)
    exp = aiplatform.Experiment(experiment_name)
    tb = exp.get_backing_tensorboard_resource()
    if not tb:
        tb = aiplatform.Tensorboard.create()
        exp.assign_backinig_tensorboard(tb)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    run_id = 'run-' + timestamp
    logging.info('Initialize the experiment, run=' + run_id)
    with aiplatform.start_run(run_id) as run:
        data_params = {'train_csv': train_csv, 'test_ratio': test_ratio}
        run.log_params(data_params)
        model_params = {'window': window, 'epoch': epoch, 'batch': batch}
        run.log_params(model_params)
        log_dir = 'logs'

        stock_dates, stock_prices = load_data_csv(train_csv)
        scaler, scaled_data = transform_data(stock_prices)
        train_split = int(len(scaled_data) * (1.0 - test_ratio))
        scaled_data_train = scaled_data[:train_split]
        x_train, y_train = extract_x_y(
            scaled_data_train, window=window, offset=window)
        model = build_LSTM(x_train, units=window)

        logging.info('Train model')
        history = train_model(model, x_train, y_train, epoch, batch,
                              interactive_progress=True, tensorboard_path=log_dir)
        run.log_params(history.params)
        for i in range(0, history.params['epochs']):
            run.log_time_series_metrics({'train_loss': history.history['loss'][i]})
        
        # logging.info('Evaluate model')
        # x_test, y_test = extract_x_y(
        #     scaled_data, window=window, offset=train_split)
        # test_loss, test_accuracy = model.evaluate(x_test, y_test)
        # run.log_params({'test_loss': test_loss})

        # save_scaler(scaler, SCALER_PATH)
        # save_model(model, MODEL_PATH)
        
        logging.info('Exit the run')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    arg_vars = vars(get_args())
    kwargs = {k: arg_vars[k] for k in
              ('experiment_name', 'project_id', 'location',
               'train_csv',
               'window', 'epoch', 'batch', 'test_ratio')}
    main(**kwargs)
