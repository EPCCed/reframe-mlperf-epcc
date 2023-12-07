import reframe as rfm
import reframe.utility.sanity as sn
from base import DeepCamBaseCheck

@rfm.simple_test
class DeepCamCPUCheck(DeepCamBaseCheck):
    def __init__(self):
        super().__init__()
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.descr = "DeepCam CPU Benchmark"
        self.valid_systems = ['archer2:compute', 'cirrus:compute']
        if self.current_system.name in ["archer2"]:
            self.num_tasks = 32
            self.num_task_per_node=1
            self.num_cpus_per_task = 128
            self.time_limit = "1h"
            self.env_vars = {
                'OMP_NUM_THREADS': str(self.num_cpus_per_task),
                "SRUN_CPUS_PER_TASK" : str(self.num_cpus_per_task)
            }
            self.prerun_cmds = ['eval "$(/work/ta127/ta127/chrisrae/miniconda3/bin/conda shell.bash hook)"', "conda activate mlperf-torch"]
            self.executable = 'python'
            self.executable_opts = ["/work/ta127/ta127/chrisrae/chris-ml-intern/ML_HPC/DeepCAM/Torch/train.py", 
                                    " --config", "/work/ta127/ta127/chrisrae/chris-ml-intern/ML_HPC/DeepCAM/Torch/configs/archer2benchmark_config.yaml"
                                    ]

            #self.reference = {"archer2:compute": {"Throughput": (200, -0.05, 0.10, "images/s"),
            #                                      "Communication Time": (295, -0.1, 0.1, "s")}
            #                 }

        elif self.current_system.name in ["cirrus"]:
            self.modules = ["openmpi/4.1.5"]
            self.num_tasks = 32
            self.num_task_per_node=1
            self.num_cpus_per_task = 36
            self.time_limit = "1h"
            self.env_vars = {
                'OMP_NUM_THREADS': str(self.num_cpus_per_task),
                "OMPI_MCA_mpi_warn_on_fork": 0
            }
            self.prerun_cmds = ['eval "$(/work/z043/z043/crae/miniconda3/bin/conda shell.bash hook)"', "conda activate mlperf-torch"]
            self.executable = 'python'
            self.executable_opts = ["/work/z043/z043/chrisrae/chris-ml-intern/ML/ResNet50/Torch/train.py",
                                    " --config", "/work/ta127/ta127/chrisrae/chris-ml-intern/ML/ResNet50/Torch/configs/cirrusbenchmark_config.yaml"
                                    ]
    
    @run_before('run')
    def set_task_distribution(self):
        self.job.options = ['--distribution=block:block']
        self.job.options = ['--account=ta127-chrisrae']
    

    


