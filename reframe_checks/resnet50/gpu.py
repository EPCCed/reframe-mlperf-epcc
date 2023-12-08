import reframe as rfm
import reframe.utility.sanity as sn
from base import ResNet50BaseCheck

@rfm.simple_test
class ResNet50GPUBenchmark(ResNet50BaseCheck):
    def __init__(self):
        super().__init__()
        self.valid_prog_environs = ["*"]
        self.descr = "ResNet50 GPU Benchmark"
        self.valid_systems = ['cirrus:compute-gpu']
        self.modules = ["openmpi/4.1.5-cuda-11.6"]
        self.num_tasks = None
        self.extra_resources = {
            "qos": {"qos": "gpu"},
            "gpu": {"num_gpus_per_node": "4"},
        }
        self.time_limit = "1h"
        self.env_vars = {
            'OMP_NUM_THREADS': "5",
            "SRUN_CPUS_PER_TASK" : "5",
            "OMPI_MCA_mpi_warn_on_fork": "0",
            "PARAMS": '"--ntasks=4 --tasks-per-node=4"' #  --nodes=8
        }
        self.prerun_cmds = ['eval "$(/work/z043/z043/crae/miniconda3/bin/conda shell.bash hook)"', 
                            "conda activate mlperf-torch", 
                            "module unload nvidia/nvhpc",
                            #"salloc --exclusive --nodes=8 --time=00:30:00 --gres=gpu:4 --partition=gpu --qos=gpu --account=z043"
                            ]
        self.executable = 'python'
        self.executable_opts = ["/work/z043/z043/crae/chris-ml-intern/ML/ResNet50/Torch/train.py",
                                "--config", "/work/z043/z043/crae/chris-ml-intern/ML/ResNet50/Torch/configs/cirrusfull_config.yaml",
                                "--device", "cuda"
        ]
        #self.postrun_cmds = ["exit"]  if salloc worked

    
    @run_before('run')
    def set_task_distribution(self):
        self.job.options = ["--exclusive"]


    @run_after("setup")
    def setup_gpu_options(self):
        """sets up different resources for gpu systems"""
        self.job.launcher.options.append("${PARAMS}")

        
        
