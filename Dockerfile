FROM myelintek/pytorch-gpu:20.06.01


ENV SCALER_PATH=/working/scaler.pkl
ENV PREDICT_ENDPOINT=http://localhost:5050/v1/models/stock_pred

RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | \
  tee /etc/apt/sources.list.d/tensorflow-serving.list && \
  curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add - &&\
  apt-get update && \
  apt-get install tensorflow-model-server && \ 
  mkdir -p /working/ && \
  wget https://dl.minio.io/client/mc/release/linux-amd64/mc && \
  chmod +x mc && mv ./mc /usr/local/bin/mc && \
  /usr/local/bin/mc config host add ms3 https://s3.myelintek.com readonly_user password && \
  /usr/local/bin/mc mirror --overwrite ms3/stock-pred-model /working/

COPY requirements.txt requirements_web.txt train.py web.py train_lstm.ipynb mlsteam.yml ./
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh && \
  pip install -r requirements.txt && \
  pip install -r requirements_web.txt 

# model path for inference
VOLUME ["/working/"]
EXPOSE 8501/tcp
EXPOSE 5050/tcp
#cmd for
#   lab  jupytor
#   track  iptyhon train_lstm.ipynb
#   action  ipython train_lstm.ipynb
#   webapp   streamlit run web.py --server.headless trure --server.enableWebsocketCompression=false
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["streamlit", "run", "web.py", "--server.headless=true", "--server.enableWebsocketCompression=false"]
