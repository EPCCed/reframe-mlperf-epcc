#!/bin/bash
#SBATCH --job-name=mlperf-CosmoFlow        # Job name
#SBATCH --tasks-per-node=1        # There is only one node on SDF-CS1
#SBATCH --cpus-per-task=128        # Each cpu is a core
#SBATCH --gres=cs:1               # Request CS-2 system
#SBATCH --output=example_%j.log   # Standard output and error log
#SBATCH --time=01:00:00           # Set time limit for this job to 1 hour

export PATH=$PATH:/home/y26/shared/bin

csrun_wse python-pt run.py --mode train --cs_ip 172.24.102.121:9000 --params params.yaml
