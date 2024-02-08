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

# Build Pytorch
```bash
srun --nodes=1 --time=01:30:00 --partition=standard --qos=standard --account=[CODE] --pty /usr/bin/bash --login
source $PREFIX/miniconda/bin/activate mlperf-torch
module load PrgEnv-gnu
cd /path/to/pytorch
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export USE_CUDA=0
export USE_ROCM=0
export USE_DISTRIBUTED=1
export BUILD_CAFFE2=0
export BUILD_CAFFE2_OPS=0
export BUILD_TEST=0
python setup.py develop
```

# Build TorchVision
```bash
git clone --single-branch --branch release/0.15 https://github.com/pytorch/vision.git
conda install libpng libjpeg-turbo
pip install expecttest flake8 typing mypy pytest pytest-mock scipy pillow
srun --exclusive --nodes=1 --time=01:30:00 --partition=gpu --qos=gpu --gres=gpu:1 --account=[CODE] --pty /usr/bin/bash --login
source $PREFIX/miniconda/bin/activate mlperf-torch
module unload cmake
cd vision
python setup.py develop
```
