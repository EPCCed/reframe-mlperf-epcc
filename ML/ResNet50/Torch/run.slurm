#!/bin/bash

#SBATCH --job-name=mlperf-resnet-benchmark
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=128

#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=ta127-chrisrae

eval "$(/work/ta127/ta127/chrisrae/miniconda3/bin/conda shell.bash hook)"
conda activate mlperf-torch

export OMPI_MCA_mpi_warn_on_fork=0

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=128

srun --hint=nomultithread --distribution=block:block python /work/ta127/ta127/chrisrae/chris-ml-intern/ML/ResNet50/Torch/train.py
