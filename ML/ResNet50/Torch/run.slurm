#!/bin/bash

#SBATCH --job-name=mlperf-resnet-benchmark
#SBATCH --time=02:00:00
#SBATCH --nodes=16
#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=4

#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=ta127-chrisrae

eval "$(/work/ta127/ta127/chrisrae/miniconda3/bin/conda shell.bash hook)"
conda activate mlperf-torch

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=4

srun --hint=nomultithread --distribution=block:block python /work/ta127/ta127/chrisrae/chris-ml-intern/ML/ResNet50/Torch/train.py