# Train.py

 Each benchmark has a custom implementation of the training loop which is defined in train.py. The reason for this is so that the benchmarks can be easily ported over to the cerrebras and graphcore systems without a drastic change in the underlying code allowing a more fair comparison.

 # Launching Jobs

 Each benchmark can be run on cirrus and archer2 using `srun [...] python train.py` as the train script is designed to work with an mpi distributed backend. In order to use the mpi backend make sure pytorch is built from source as shown in `HOW_TO/build_torch` and within the your slurm launch script make your you activate the conda environment.

An example launch script may look like this:

```bash 
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

module load openmpi/4.1.5-11.6  # only needed on cirrus

export OMPI_MCA_mpi_warn_on_fork=0

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=36

srun --hint=nomultithread --distribution=block:block python /work/z043/z043/crae/chris-ml-intern/ML/ResNet50/Torch/train.py
```

Its important to note that mpi is not loaded by default on cirrus so must by loaded with `module load openmpi/4.1.5-11.6`.

You can easily increase the number of nodes by changing the  `#SBATCH --nodes=...` line.


