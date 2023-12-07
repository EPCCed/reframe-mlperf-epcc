import reframe.utility.sanity as sn
from base import DeepCamBaseCheck

class CosmoFlowGPUBenchmark(DeepCamBaseCheck):
    def __init__(self):
        super().__init__()
        self.valid_prog_environs = ['*']
        self.descr = "DeepCam CPU Benchmark"
        self.valid_systems = ['cirrus:compute-gpu']

        
        self.modules = ["openmpi/4.1.5-cuda-11.6"]
        self.num_tasks = 32
        self.num_task_per_node=4
        self.num_cpus_per_task = 9
        self.extra_resources = {
            "qos": {"qos": "gpu"},
            "gpu": {"num_gpus_per_node": "4"},
        }
        self.time_limit = "1h"
        self.env_vars = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            "OMPI_MCA_mpi_warn_on_fork": 0
        }
        self.prerun_cmds = ['eval "$(/work/z043/z043/crae/miniconda3/bin/conda shell.bash hook)"', "conda activate mlperf-torch"]
        self.executable_opts = ["/work/z043/z043/crae/chris-ml-intern/ML_HPC/CosmoFlow/Torch/train.py",
                                "--config", "/work/z043/z043/crae/chris-ml-intern/ML_HPC/CosmoFlow/Torch/configs/cirrusbenchmark_config.yaml",
                                "--device", "cuda"
        ]
