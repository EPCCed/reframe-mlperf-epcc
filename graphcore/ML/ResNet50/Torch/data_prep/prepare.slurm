#!/bin/bash

#SBATCH --job-name=prepare-imagenet
#SBATCH --time=00:10:00
#SBATCH --nodes=4
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --account=ta127-chrisrae

eval "$(/work/ta127/ta127/chrisrae/miniconda3/bin/conda shell.bash hook)"
conda activate mlperf-torch

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1

srun  --hint=nomultithread --distribution=block:block python /work/ta127/ta127/chrisrae/chris-ml-intern/ML/ResNet50/Torch/data_prep/prepare.py --clean=true