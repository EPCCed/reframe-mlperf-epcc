#!/bin/bash

#SBATCH --job-name=mlperf-resnet-benchmark
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=16

#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=z043

eval "$(/work/z043/z043/crae/miniconda3/bin/conda shell.bash hook)"
conda activate mlperf-torch

module load openmpi/4.1.4

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=16

srun --hint=nomultithread --distribution=block:block python /work/z043/z043/crae/chris-ml-intern/ML/ResNet50/Torch/train.py