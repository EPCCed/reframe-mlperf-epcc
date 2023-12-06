# Setup Miniconda

```bash
export $PREFIX=/work/ta127/ta127/chrisrae/
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
#When installing make sure you install miniconda in the correct PREFIX 
#Make sure that miniconda is installed in /work not /home
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

# Install Dependicies
```bash
source $PREFIX/miniconda/bin/activate mlperf-torch
conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
conda install -c nvidia cudatoolkit=11.6
conda install -c nvidia cudnn
conda install -c "nvidia/label/broken" nccl
conda install -c "nvidia/label/cuda-11.6.0" libcusparse-dev
conda install -c "nvidia/label/cuda-11.6.0" libcusolver-dev
conda install cmake ninja
#cd into pytorch if not already
pip install -r requirements.txt
conda install mkl mkl-include
conda install -c pytorch magma-cuda116+
```

# Build Pytorch
```bash
srun --exclusive --nodes=1 --time=01:30:00 --partition=gpu --qos=gpu --gres=gpu:1 --account=[CODE] --pty /usr/bin/bash --login
source $PREFIX/miniconda/bin/activate mlperf-torch
export USE_ROCM=0
export BUILD_TEST=0
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_NVCC_EXECUTABLE=$CONDA_PREFIX/bin/nvcc
export CUDA_BIN_PATH=$CONDA_PREFIX/bin
export CUDA_HOME=$CONDA_PREFIX
export LDFLAGS=-L/$CONDA_PREFIX/lib
module unload cmake
module swap gcc gcc/10.2.0
module load openmpi/4.1.5-cuda-11.6
#cd into pytorch if not already
python setup.py develop
```

## Build torchvision (for resnet)
```bash
git clone --single-branch --branch release/0.15 https://github.com/pytorch/vision.git
conda install libpng libjpeg-turbo
srun --exclusive --nodes=1 --time=01:30:00 --partition=gpu --qos=gpu --gres=gpu:1 --account=[CODE] --pty /usr/bin/bash --login
source $PREFIX/miniconda/bin/activate mlperf-torch
module unload cmake
module swap gcc gcc/10.2.0
cd vision
python setup.py develop
```