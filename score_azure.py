import json
import logging
import os

import numpy as np

from train import load_model, load_scaler, extract_x_y


model = None
scaler = None
WINDOW = 50


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model, scaler

    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    model_base_dir = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'trained')
    model_path = os.path.join(model_base_dir, 'model')
    scaler_path = os.path.join(model_base_dir, 'scaler.pkl')

    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    """
    logging.info("model: request received")
    data = np.array(json.loads(raw_data)["data"])
    if len(data) < 50:
        logging.warn(f'Insufficient stock prices (currently {len(data)}) given. '
                     f'Please provide at least {WINDOW} records.')
        raise ValueError('insufficient stock precies given')
    
    scaled_data = scaler.transform(data[:,None])
    x_scaled, y_scaled = extract_x_y(scaled_data + np.array([.0]), window=WINDOW, offset=WINDOW)
    y_pred_scaled = model.predict(x_scaled)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    logging.info("Request processed")
    return y_pred.tolist()
