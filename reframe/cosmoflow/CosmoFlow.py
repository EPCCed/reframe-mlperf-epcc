import reframe as rfm
import reframe.utility.sanity as sn

@rfm.simple_test
class Cosmoflow(rfm.RegressionTest):
    def __init__(self):
        self.descr = 'CosmoFlow Benchmark'
        self.valid_systems = ['archer2:compute'] 
        self.valid_prog_environs = ["*"]

        num_nodes = 16
        self.time_limit = '2h'
        self.num_tasks_per_node = 4
        self.num_tasks = num_nodes*self.num_tasks_per_node
        self.num_cpus_per_task = 32
        self.extra_resources = {'qos': {'qos': 'standard'}}

        self.prerun_cmds = ['eval "$(/work/ta127/ta127/chrisrae/miniconda3/bin/conda shell.bash hook)"', "conda activate mlperf-torch"]
        self.executable = 'python'
        self.executable_opts = ["/work/ta127/ta127/chrisrae/chris-ml-intern/ML_HPC/CosmoFlow/Torch/train.py"]

        # Set up performance logging
        ...