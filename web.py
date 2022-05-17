import os
import pickle
from io import StringIO

import numpy as np
import streamlit as st

import train


WINDOW = 50


def get_scaler():
    pkl_file_path = os.environ['SCALER_PATH']
    return train.load_scaler(pkl_file_path)


def read_file(content: StringIO):
    raw_data = np.genfromtxt(content, dtype=[float,])
    return raw_data


def preproc_data(raw_data, scaler):
    scaled_data = scaler.transform(raw_data[:,None])
    x_history, y_history = train.extract_x_y(scaled_data, window=WINDOW, offset=WINDOW)
    x_predict = train.extract_x_y(scaled_data + np.array([.0]), window=WINDOW, offset=len(scaled_data) - WINDOW + 1)
    return x_history, y_history, x_predict[0:1]


def main():
    st.title('Stock Market Prediction')
    
    try:
        scaler = get_scaler()
    except:
        st.error('Failed to load scaler. Make sure $SCALER_PATH is set and the scaler file exists in that path.')
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
        x_history, y_history, x_predict = preproc_data(data, scaler)
        st.write(len(x_history))
        st.write(y_history)
        st.write(len(x_predict))


main()
