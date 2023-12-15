# Setup Miniconda

```bash
export $PREFIX=/work/ta127/ta127/chrisrae/
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
#When installing make sure you install miniconda in the correct PREFIX 
#Make sure that miniconda is install in /work not /home
eval "$($PREFIX/miniconda3/bin/conda shell.bash hook)"
conda create --name mlperf-torch python=3.10
```

# Download Torch
```bash
git clone --single-branch --branch release/2.0 https://github.com/pytorch/pytorch.git
cd pytorch
git submodule sync
git submodule update --init --recursive
```

# Install dependencies
```bash
source $PREFIX/miniconda/bin/activate mlperf-torch
conda install cmake ninja
cd ./pytorch
pip install -r requirements.txt
conda install mkl mkl-include
```

# Define *build.slurm*

This could be done with an interactive node

```bash
#!/bin/bash (build.slurm)

#SBATCH --job-name=build-torch
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --account=[Budget Code]

#change to your prefix
eval "$(/work/ta127/ta127/chrisrae/miniconda3/bin/conda shell.bash hook)"
conda activate mlperf-torch

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1

module load PrgEnv-gnu

cd /work/ta127/ta127/chrisrae/pytorch

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export USE_CUDA=0
export USE_ROCM=0
export USE_DISTRIBUTED=1
export BUILD_CAFFE2=0
export BUILD_TEST=0

python setup.py develop
```

# launch slurm job
```bash
sbatch build.slurm
```
                                                    