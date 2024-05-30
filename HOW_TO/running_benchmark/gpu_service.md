# Running Benchmarks

## ReFrame (easiest)

The easiest way to run the benchmarks with the pre-defined reframe test.

First clone the forked epcc-reframe repo:

```bash 
git clone https://github.com/BigBalloon8/epcc-reframe.git
```

The EIDF uses kubernetes to launch and manage jobs. By default reframe does not support kubernetes, we have our own implementation of reframe with k8s support. To use it install reframe on your VM:

```bash
git clone --single-branch –branch k8s https://github.com/BigBalloon8/reframe.git 
cd reframe
pip install -e .
```

then its as easy as follows:
```bash
export REFRAME_CONFIG=/path/to/repo/epcc-reframe/configurations/eidf_settings.py
cd epcc-reframe/tests/mlperf
```

If you want to run a single test for example resnet5 you can run:

```bash 
reframe -C ${REFRAME_CONFIG} -c ./resnet50/eidf_gpu.py -r --performance-report
```

To run all the test you can run:

```bash 
# asssuming your in /path/to/repo/epcc-reframe/tests/mlperf
reframe -C ${REFRAME_CONFIG} -c . -R -r --performance-report
```

## kubectl ... (easy)
As k8s runs containerized application when making changes to the code in this repo you must re-build the container before running it again.

If you want to run the scripts outside of a reframe test that is also possible. All of the code is available in this repo.

I will use resnet50 as an example but all the benchmarks follow the same process (note only cosmoflow, deepcam, and resnet50 currently work).

first clone this repo if not already done and cd into `ML/ResNet50/Torch`:

```bash
git clone https://github.com/EPCCed/reframe-mlperf-epcc.git
cd reframe-mlperf-epcc/ML/ResNet50/Torch
```

Within this dir you will find [`train.py`](../../ML/ResNet50/Torch/train.py), [`config.yaml`](../../ML/ResNet50/Torch/config.yaml) and the `configs/` directory, these are most likely the files you want to change unless you want to customize the model, dataloader or optimizer which can be found in their respective files/dirs within `ML/ResNet50/Torch` (Its fairly intuitive). Explanations of what different variables within the config do can be found in [`HOW_TO/running_benchmark/config_explanation.md`](./config_explanation.md)

After any changes are made the following commands must be called:
```bash
cd reframe-mlperf-epcc/
sudo docker build -t=.../mlperf-epcc:latest .
sudo docker push .../mlperf-epcc 
```

Then you need a Pod config that describes all the information associated it:

```yaml
#mlperf_pod.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    kueue.x-k8s.io/queue-name: ...-user-queue
  name: mlperf-resnet
spec:
  containers:
  - args:
    - --nproc_per_node=4
    - /workspace/ML/ResNet50/Torch/train.py
    - -lbs
    - '8'
    - -c
    - /workspace/ML/ResNet50/Torch/config.yaml
    - --t_subset_size
    - '2048'
    - --v_subset_size
    - '512'
    command:
    - torchrun
    env:
    - name: OMP_NUM_THREADS
      value: '4'
    image: .../mlperf-epcc
    name: mlperf-resnet
    resources:
      limits:
        cpu: 16
        memory: 32Gi
        nvidia.com/gpu: 4
      requests:
        cpu: 16
        memory: 16Gi
    volumeMounts:
    - mountPath: /mnt/ceph_rbd
      name: volume
    - mountPath: /dev/shm
      name: devshm
    workingDir: /workspace/ML
  nodeSelector:
    nvidia.com/gpu.product: NVIDIA-A100-SXM4-40GB
  restartPolicy: Never
  volumes:
  - name: volume
    persistentVolumeClaim:
      claimName: imagenet-pv
  - emptyDir:
      medium: Memory
    name: devshm
```
We expose certain variables that can be changed through the command line (these are also available on cirrus and a2 as they use the same source code):
```
$ python train.py --help
Usage: train.py [OPTIONS]

Options:
  -d, --device TEXT               The device type to run the benchmark on
                                  (cpu|gpu|cuda). If not provided will default
                                  to config.yaml
  -c, --config TEXT               Path to config.yaml. If not provided will
                                  default to config.yaml in the cwd  [default:
                                  /home/eidf095/eidf095/crae-ml/reframe-
                                  mlperf-epcc/config.yaml]
  --data-dir TEXT                 Path To ResNet50 dataset. If not provided
                                  will default to what is provided in the
                                  config.yaml
  -gbs, --global_batchsize INTEGER
                                  The Global Batchsize
  -lbs, --local_batchsize INTEGER
                                  The Local Batchsize, Leave as 0 to use the
                                  Global Batchsize
  --t_subset_size INTEGER         Size of the Training Subset, dont call to
                                  use full dataset
  --v_subset_size INTEGER         Size of the Validation Subset, dont call to
                                  use full dataset
  --help                          Show this message and exit.

```
To run the Pod run:
```bash
kubectl create -f mlperf_pod.yaml
kubectl logs -f mlperf-resnet
kubectl delete pod mlperf-resnet
```
