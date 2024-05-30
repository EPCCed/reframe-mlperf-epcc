# Running benchmarks on Cerebras

create a python environment

```bash
python3.8 -m venv mlperf_cs2_pt
source mlperf_cs2_pt
pip install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install cerebras_pytorch==2.1.1
```
clone this source code
```bash
git clone https://github.com/EPCCed/reframe-mlperf-epcc.git
cd reframe-mlperf-epcc
```
create your slurm batch script
```bash
#!/bin/bash
#SBATCH --job-name=mlperf-ResNet        # Job name
#SBATCH --cpus-per-task=16         # Request 16 cores
#SBATCH --time=12:00:00           # Set time limit for this job to 12 hour
#SBATCH --gres=cs:1


source mlperf_cs2_pt/bin/activate

python reframe-mlperf-epcc/cs2/ML/ResNet50/train.py
```

run your batch script
```bash
sbatch run.slurm
```