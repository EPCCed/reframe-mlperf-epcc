# Running Benchmarks

## ReFrame (easiest)

The easiest way to run the benchmarks with the pre-defined reframe test.

First clone the forked epcc-reframe repo:

```bash 
git clone https://github.com/BigBalloon8/epcc-reframe.git
```

All of the python scrips, configs and miniconda environments have been setup in predefined locations on both archer2 and cirrus (note at the time of writing this pytorch needs to be built from source on archer2 so the reframe tests wont run on archer2 yet). 

then its as easy as follows:
```bash
module load reframe epcc-reframe
# Pick the correct configuration for the system (cirrus_settings.py | archer2_settings.py)
export REFRAME_CONFIG=/path/to/repo/epcc-reframe/configurations/..._settings.py
cd epcc-reframe/tests/mlperf
```

If you want to run a single test for example resnet50 on a gpu you can run:

```bash 
reframe -C ${REFRAME_CONFIG} -c ./resnet50/gpu.py -r --performance-report
```

To run all the test you can run:

```bash 
# asssuming your in /path/to/repo/epcc-reframe/tests/mlperf
reframe -C ${REFRAME_CONFIG} -c . -R -r --performance-report
```

If you want to customize your runs by changing the configs or train scrips you can create your own configs or train scripts and point to them by changing `self.executable_opts` in the desired reframe check (e.g. `resnet50/gpu.py`) 

# srun python train.py (easy)

If you want to run the scripts outside of a reframe test that is also possible. All of the code is available in this repo. 

I will use resnet50 as an example but all the benchmarks follow the same process (note only cosmoflow, deepcam, and resnet50 currently work). 

first clone this repo if not already done and cd into `ML/ResNet50/Torch`:

```bash
git clone https://github.com/EPCCed/chris-ml-intern
cd chris-ml-intern/ML/ResNet50/Torch
```

Within this dir you will find [`train.py`](../../ML/ResNet50/Torch/train.py), [`config.yaml`](../../ML/ResNet50/Torch/config.yaml) and the `configs/` directory, these are most likely the files you want to change unless you want to customize the model, dataloader or optimizer which can be found in their respective files/dirs within `ML/ResNet50/Torch` (Its fairly intuitive). Explanations of what different variables within the config do can be found in [`HOW_TO/running_benchmark/config_explanation.md`](./config_explanation.md)

In order to change the global batch size all you have to do is change the variable within the config

To run the train.py its important to launch with srun even if you only want to test on 1 process as my implementation of distributed training uses mpi as a backend. I recommend creating a launch script rather than using srun directly, this is what a launch script could look like on archer2 for a 4 node run:

```bash
#!/bin/bash

#SBATCH --job-name=mlperf-resnet-benchmark
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=128

#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=ta127-chrisrae

eval "$(/work/ta127/shared/miniconda3/bin/conda shell.bash hook)"
conda activate mlperf-torch

export OMPI_MCA_mpi_warn_on_fork=0

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=128

srun --hint=nomultithread --distribution=block:block \
        python /work/ta127/ta127/chrisrae/chris-ml-intern/ML/ResNet50/Torch/train.py \
        --config /work/ta127/ta127/chrisrae/chris-ml-intern/ML/ResNet50/Torch/config.yaml
```

Make sure you point the --config to the correct config, if you want to increase the number of nodes to 8 all you would set `#SBATCH --nodes=8`. 

There are a few different ways to run the job on gpus, first you can change the the `device` option in the config to `gpu` or `cuda` alternatively you can add the `--device cuda` argument before or after `--config /path/to/config` the latter will allow you to use the same config for cpu and gpu jobs. Here is an example 16 gpu launch script on cirrus could look ike:

```bash
#!/bin/bash

#SBATCH --job-name=mlperf-resnet-benchmark
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --nodes=4 #MAKE SURE TO UPDATE SRUN

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --account=z043

eval "$(/work/z043/shared/miniconda3/bin/conda shell.bash hook)"
conda activate mlperf-torch

module load openmpi/4.1.5-cuda-11.6

export OMPI_MCA_mpi_warn_on_fork=0

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=9

srun --ntasks=16 --tasks-per-node=4 \
        python /work/z043/z043/crae/chris-ml-intern/ML/ResNet50/Torch/train.py \
        --config /work/ta127/ta127/chrisrae/chris-ml-intern/ML/ResNet50/Torch/config.yaml\
        --device cuda
```

When changing the number of nodes its important to update the ntasks `srun --ntasks=(num_nodes*gpus_per_node)` its also important to note when running job on cirrus you have to load mpi yourself with `module load openmpi/4.1.5-cuda-11.6` the version of mpi is important so dont just load openmpi.

If you want to try using libraries that are not in the shared miniconda such as nvidias dali to test faster data loading, you will have to setup your own miniconda env, make sure you build pyotrch from source so that it works with an mpi distributed backend, the details on how to do so can be found at in this repo at [`HOW_TO/build_torch`](../build_torch).
