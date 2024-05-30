# whereis 

###  DATA

| Service | ImageNet | CosmoFlow | DeepCAM |
| :-------: | :-------- | :--------- | :------- |
| Archer2 | `/work/ta127/shared/imagenet-1k/data/` | `/work/z04/shared/mlperf-hpc/cosmoflow/cosmoUniverse_2019_05_4parE_tf_v/` | `/work/z04/shared/mlperf-hpc/deepcam/mini/deepcam-data-n512` 
| Cirrus | `/work/z043/shared/imagenet-1k/huggingface/data/` | `/work/z04/shared/mlperf-hpc/cosmoflow/cosmoUniverse_2019_05_4parE_tf_v2/` | `/work/z04/shared/mlperf-hpc/deepcam/mini/` |
| EIDF-GPU | `imagenet-pv:/imagenet-1k/data`/ | `cosmoflow-pvc:/cosmoUniverse_2019_05_4parE_tf_v2/` | `deepcam-mini:/deepcam/mini/deepcam-data-n512/`  OR  `deepcam-pvc:/gridftp-save/deepcam/All-Hist/`|
| Cerebras | `/mnt/e1000/home/z043/z043/crae-cs1/chris-ml-intern/cs2/ML/ResNet50/data/` | N/A | N/A |
| GraphCore | `imagenet-pvc:imagenet-1k/data/` | `cosmoflow-pvc:/cosmoUniverse_2019_05_4parE_tf_v2/` | N/A |

All graphcore PVCs are found under the graphcore context found in namespace `eidf095ns` run `kubectl get --context=graphcore pvc` to view.

### Miniconda envs

| Service | Location | envs |
| :-------: | :-------- | :--------- |
| A2 | /work/z043/shared/miniconda3/ | mlperf-torch-cpu, mlperf-torch-rocm |
| Cirrus | /work/z043/shared/miniconda3/ | mlperf-torch |
|EIDF VM| /home/eidf095/shared/miniconda3/ | torch-cuda |

### Containers
| Service | Container |
| :-------: | :-------- |
|GPU-service| https://hub.docker.com/repository/docker/bigballoon8/mlperf-epcc/general |
|GraphCore  | https://hub.docker.com/repository/docker/bigballoon8/mlperf-epcc-gc/general |



