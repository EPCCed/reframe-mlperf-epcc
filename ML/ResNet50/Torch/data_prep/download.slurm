#!/bin/bash

#SBATCH --job-name=download-imagenet
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --partition=serial
#SBATCH --qos=serial
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=ta127-chrisrae

eval "$(/work/ta127/ta127/chrisrae/miniconda3/bin/conda shell.bash hook)"
conda activate mlperf-torch

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1

curl --output /work/ta127/shared/imagenet-1k/val/ILSVRC2012_img_val.tar https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
curl --output /work/ta127/shared/imagenet-1k/test/ILSVRC2012_img_test_v10102019.tar https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar
