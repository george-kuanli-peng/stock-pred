#!/bin/bash

if [ -e "$SCALER_PATH" ] && [[ $PREDICT_ENDPOINT == http://localhost* ]]; then
  tensorflow_model_server --rest_api_port=5050 --model_name=stock_pred --model_base_path=/models/ &
fi

exec "$@"