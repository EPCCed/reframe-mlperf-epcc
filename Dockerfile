FROM nvcr.io/nvidia/pytorch:23.04-py3

RUN pip install torchmetrics
RUN pip install "git+https://github.com/mlperf/logging.git"
RUN pip install tfrecord
RUN pip install h5py
RUN pip install torchdata


COPY . .


