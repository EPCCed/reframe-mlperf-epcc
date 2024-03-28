#!/bin/bash
#SBATCH --job-name=mlperf-DeepCAM        # Job name
#SBATCH --cpus-per-task=16        # Each cpu is a core
#SBATCH --gres=cs:1
#SBATCH --time=06:00:00           # Set time limit for this job to 6 hour

source /home/z043/z043/crae-cs1/mlperf_cs2_pt/bin/activate

python train.py
