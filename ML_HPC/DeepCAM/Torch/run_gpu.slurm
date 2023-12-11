#!/bin/bash

#SBATCH --job-name=mlperf-deepcam-benchmark
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --nodes=4 #MAKE SURE TO UPDATE SRUN

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --account=z043

eval "$(/work/z043/z043/crae/miniconda3/bin/conda shell.bash hook)"
conda activate mlperf-torch

module load openmpi/4.1.5-cuda-11.6

export OMPI_MCA_mpi_warn_on_fork=0

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=5

srun --ntasks=16 --tasks-per-node=4 python /work/z043/z043/crae/chris-ml-intern/ML_HPC/DeepCAM/Torch/train.py --device cuda --config /work/z043/z043/crae/chris-ml-intern/ML_HPC/DeepCAM/Torch/configs/cirrusbenchmark_config.yaml