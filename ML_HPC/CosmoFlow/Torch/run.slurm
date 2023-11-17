#!/bin/bash

#SBATCH --job-name=mlperf-cosmoflow-benchmark
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=32

#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=ta127-chrisrae

eval "$(/work/ta127/ta127/chrisrae/miniconda3/bin/conda shell.bash hook)"
conda activate mlperf-torch

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=32

srun --hint=nomultithread --distribution=block:block python /work/ta127/ta127/chrisrae/chris-ml-intern/ML_HPC/CosmoFlow/Torch/train.py