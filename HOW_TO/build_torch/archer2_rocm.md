```bash
export PREFIX=/work/ta127/ta127/chrisrae/
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
#When installing make sure you install miniconda in the correct PREFIX
#Make sure that miniconda is install in /work not /home
eval "$($PREFIX/miniconda3/bin/conda shell.bash hook)"
conda create --name mlperf-torch-amd python=3.10
```

# Download Torch
```bash
git clone --single-branch --branch release/2.2 https://github.com/pytorch/pytorch.git
cd pytorch
git submodule sync
git submodule update --init --recursive
https://github.com/ROCm/ROCm/issues/2121
```

# Install dependencies
```bash
source $PREFIX/miniconda/bin/activate mlperf-torch
conda install cmake ninja
cd ./pytorch
pip install -r requirements.txt
conda install mkl mkl-include
conda install anaconda::ncurses
conda install -c conda-forge ncurses
```

# Build Pytorch
```bash
srun --gpus=1 --time=01:00:00 --partition=gpu --qos=gpu-shd --account=[CODE] --pty /bin/bash
source $PREFIX/miniconda/bin/activate mlperf-torch
module load PrgEnv-gnu
module load rocm
module load craype-accel-amd-gfx90a
module load craype-x86-milan
cd /path/to/pytorch
export MPICH_GPU_SUPPORT_ENABLED=1
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export LD_LIBRARY_PATH=$CRAY_MPICH_ROOTDIR/gtl/lib/:$LD_LIBRARY_PATH
export LD_PRELOAD=$CRAY_MPICH_ROOTDIR/gtl/lib/libmpi_gtl_hsa.so:$LD_PRELOAD
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$CONDA_PREFIX/bin:$PATH
export USE_CUDA=0
export USE_ROCM=1
export USE_DISTRIBUTED=1
export BUILD_CAFFE2=0
export BUILD_TEST=0
export PYTORCH_ROCM_ARCH=gfx90a
export BUILD_CAFFE2_OPS=0
python tools/amd_build/build_amd.py
python setup.py develop
# to re-build run: python setup.py clean 
```

```bash
git clone --single-branch --branch release/0.15 https://github.com/pytorch/vision.git
source $PREFIX/miniconda/bin/activate mlperf-torch
conda install libpng libjpeg-turbo
pip install expecttest flake8 typing mypy pytest pytest-mock scipy
srun --gpus=1 --time=01:00:00 --partition=gpu --qos=gpu-shd --account=[CODE] --pty /bin/bash
source $PREFIX/miniconda/bin/activate mlperf-torch
module load PrgEnv-gnu
module load rocm
module load craype-accel-amd-gfx90a
module load craype-x86-milan
module unload cmake
cd vision
python setup.py develop
```