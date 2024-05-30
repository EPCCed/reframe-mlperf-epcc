# Running benchmarks on graphcore

Start by cloning this repo and building the container:

```bash
git clone https://github.com/EPCCed/reframe-mlperf-epcc.git
cd reframe-mlperf-epcc/graphcore
sudo docker build -t=.../mlperf-epcc-gc:latest .
sudo docker push .../mlperf-epcc-gc
```

Then Create your IPUJob config:

```yaml
#ipu_resnet.yaml
apiVersion: graphcore.ai/v1alpha1
kind: IPUJob
metadata:
  generateName: resnet-training-
spec:
  # jobInstances defines the number of job instances.
  # More than 1 job instance is usually useful for inference jobs only.
  jobInstances: 1
  # ipusPerJobInstance refers to the number of IPUs required per job instance.
  # A separate IPU partition of this size will be created by the IPU Operator
  # for each job instance.
  ipusPerJobInstance: "8"
  workers:
    template:
      spec:
        containers:
        - name: resnet-training
          image: .../mlperf-epcc-gc
          command: [/bin/bash, -c, --]
          args:
            - |
              cd /ML/ResNet50/Torch 
              python train.py -c /ML/ResNet50/Torch/config.yaml
          resources:
            limits:
              cpu: 128
              memory: 400Gi
          securityContext:
            capabilities:
              add:
              - IPC_LOCK
          volumeMounts:
          - mountPath: /dev/shm
            name: devshm
          - mountPath: /mnt/ceph_rbd
            name: volume
        restartPolicy: Never
        hostIPC: true
        volumes:
        - emptyDir:
            medium: Memory
          name: devshm
        - name: volume
          persistentVolumeClaim:
            claimName: 'imagenet-pvc'
```

```bash
kubectl create -f ipu_resnet.yaml
```
A large number of CPU cores and Memory is unusually required as the compilation jobs are often very resource intensive.

