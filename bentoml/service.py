import numpy as np
import bentoml
from bentoml.io import NumpyNdarray


loaded_model = bentoml.models.get('stock_pred_lstm:latest')
scaler = loaded_model.custom_objects['scaler']
model_runner = loaded_model.to_runner()
svc = bentoml.Service('stock_pred_lstm', runners=[model_runner])


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict_normalized(input_series: np.ndarray) -> np.ndarray:
    return model_runner.predict.run(input_series)


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_series: np.ndarray) -> np.ndarray:
    batch_size = input_series.shape[0]
    input_norm = scaler.transform(input_series.reshape(-1, 1)).reshape((batch_size, -1))
    output_norm = model_runner.predict.run(input_norm)
    print(f'input_norm: {input_norm.shape}, output_norm: {output_norm.shape}')
    return scaler.inverse_transform(output_norm)
