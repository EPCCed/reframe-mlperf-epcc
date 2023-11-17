#!/bin/bash

#SBATCH --job-name=test-slurm-envs
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --partition=serial
#SBATCH --qos=serial
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=ta127-chrisrae

eval "$(/work/ta127/ta127/chrisrae/miniconda3/bin/conda shell.bash hook)"
conda activate tfrecord

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1

srun python /work/ta127/ta127/chrisrae/chris-ml-intern/ML_HPC/CosmoFlow/Torch/data_prep/tfrecord_to_pt.py

