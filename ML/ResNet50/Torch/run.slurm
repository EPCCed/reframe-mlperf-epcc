#!/bin/bash

#SBATCH --job-name=mlperf-resnet-benchmark
#SBATCH --time=05:00:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=36

#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=z043

eval "$(/work/z043/z043/crae/miniconda3/bin/conda shell.bash hook)"
conda activate mlperf-torch

module load openmpi/4.1.5

export OMPI_MCA_mpi_warn_on_fork=0

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=36

srun --hint=nomultithread --distribution=block:block python /work/z043/z043/crae/chris-ml-intern/ML/ResNet50/Torch/train.py