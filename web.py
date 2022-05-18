import os
import pickle
from io import StringIO

import numpy as np
import requests
import streamlit as st

import train


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU due to CUDA LSTM issues

ENV_SCALER_PATH = 'SCALER_PATH'
ENV_PREDICT_ENDPOINT = 'PREDICT_ENDPOINT'
WINDOW = 50


def get_scaler():
    pkl_file_path = os.environ[ENV_SCALER_PATH]
    return train.load_scaler(pkl_file_path)


def get_predict_url():
    endpoint = os.environ[ENV_PREDICT_ENDPOINT]
    rsp = requests.get(endpoint)
    rsp.raise_for_status()
    return endpoint + ':predict'


def read_file(content: StringIO):
    raw_data = np.genfromtxt(content, dtype=[float,])
    return raw_data


def preproc_data(raw_data, scaler):
    scaled_data = scaler.transform(raw_data[:,None])
    x_history, y_history = train.extract_x_y(scaled_data, window=WINDOW, offset=WINDOW)
    x_predict, _ = train.extract_x_y(scaled_data + np.array([.0]), window=WINDOW, offset=len(scaled_data) - WINDOW + 1)
    return x_history, y_history, x_predict[0:1]


def postproc_data(scaled_data, scaler):
    post_data = scaler.inverse_transform(scaled_data)
    return post_data


def predict(predict_url, x_instances: np.array):
    # st.write(str(x_instances.tolist()))
    rsp = requests.post(predict_url, json={'instances': x_instances.tolist()})
    rsp.raise_for_status()
    return np.array(rsp.json()['predictions'])


def plot(pred, actual):
    data = np.concatenate((pred, actual), axis=1)
    st.line_chart(data)


def main():
    st.title('Stock Market Prediction')
    
    try:
        scaler = get_scaler()
    except:
        st.error(f'Failed to load scaler. '
                 f'Make sure ${ENV_SCALER_PATH} is given and the scaler file exists in that path.')
        return
    try:
        predict_url = get_predict_url()
    except:
        st.error(f'Faild to access the prediction service. '
                 f'Make sure ${ENV_PREDICT_ENDPOINT} is given a valid service url, '
                 f'such as http://192.168.0.1:5050/v1/models/stock_pred')
        return
    
    st.text('Upload stock prices in history (one closing price per line).')
    st.text(f'Prediction will start from the {WINDOW}-th record.')
    uploaded_file = st.file_uploader('Choose a file')

    if uploaded_file is not None:
        content = StringIO(uploaded_file.getvalue().decode('utf-8'))
        data = read_file(content)
        if len(data) < 50:
            st.warning(f'Insufficient stock prices (currently {len(data)}) given. '
                       f'Please provide at least {WINDOW} records, one per line.')
            return
        x_history_scaled, y_actual_history_scaled, x_predict_scaled = preproc_data(data, scaler)

        if len(x_history_scaled) > 0:
            y_predict_history_scaled = predict(predict_url, x_history_scaled)
            y_predict_history = postproc_data(y_predict_history_scaled, scaler)
            y_actual_history = postproc_data(y_actual_history_scaled, scaler)
            col1, col2 = st.columns(2)
            col1.metric(label='RMSE', value=round(train.get_rmse(pred=y_predict_history, actual=y_actual_history), 2))
            col2.metric(label='MAPE', value=round(train.get_mape(pred=y_predict_history, actual=y_actual_history), 2))
            plot(pred=y_predict_history, actual=y_actual_history)

        if len(x_predict_scaled) > 0:
            y_predict_scaled = predict(predict_url, x_predict_scaled)
            y_predict = postproc_data(y_predict_scaled, scaler)
            st.metric(label='Next prediction', value=round(y_predict[0,0], 2))


main()
