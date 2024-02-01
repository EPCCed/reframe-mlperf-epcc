FROM graphcore/pytorch:3.3.0-ubuntu-20.04-20230703

RUN pip install torchmetrics
RUN pip install "git+https://github.com/mlperf/logging.git"
RUN pip install click

COPY . .