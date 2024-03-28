#!/bin/bash
#SBATCH --job-name=mlperf-ResNet        # Job name
#SBATCH --cpus-per-task=16         # Request 2 cores
#SBATCH --time=12:00:00           # Set time limit for this job to 12 hour
#SBATCH --gres=cs:1


source /home/z043/z043/crae-cs1/mlperf_cs2_pt/bin/activate

python /home/z043/z043/crae-cs1/chris-ml-intern/cs2/ML/ResNet50/train.py
