#!/bin/bash

#SBATCH --job-name=test-slurm-envs
#SBATCH --time=00:01:00
#SBATCH --nodes=2
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=ta127-chrisrae

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1

echo $SLURM_JOB_NODELIST