eval "$(/work/ta127/ta127/chrisrae/miniconda3/bin/conda shell.bash hook)"
conda activate mlperf-torch

export OMP_NUM_THREADS=

python /work/ta127/ta127/chrisrae/chris-ml-intern/ML_HPC/CosmoFlow/Torch/train.py