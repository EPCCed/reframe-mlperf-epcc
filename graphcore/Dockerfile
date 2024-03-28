FROM graphcore/pytorch:3.3.0

RUN pip install torchmetrics
RUN pip install "git+https://github.com/mlperf/logging.git"
RUN pip install click
RUN pip install tfrecord
RUN pip install h5py
RUN pip install protobuf==3.20.0

COPY . .