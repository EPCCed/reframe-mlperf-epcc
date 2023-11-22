import reframe as rfm
import reframe.utility.sanity as sn

@rfm.simple_test
class CosmoflowCPU(rfm.RegressionTest):
    def __init__(self):
        self.descr = 'CosmoFlow Benchmark'
        self.valid_systems = ['archer2:compute', "cirrus:compute"] 
        self.valid_prog_environs = ["*"]

        num_nodes = 16
        self.time_limit = '2h'
        self.num_tasks_per_node = 4
        self.num_tasks = num_nodes*self.num_tasks_per_node
        self.num_cpus_per_task = 8
        self.extra_resources = {'qos': {'qos': 'standard'}}

        self.prerun_cmds = ['eval "$(/work/ta127/ta127/chrisrae/miniconda3/bin/conda shell.bash hook)"', "conda activate mlperf-torch"]
        self.executable = 'python'
        self.executable_opts = ["/work/ta127/ta127/chrisrae/chris-ml-intern/ML_HPC/CosmoFlow/Torch/train.py"]

    @sanity_function
    def assert_target_met():
        return sn.assert_found(r'"key": "target_mae_reached"') and sn.assert_found(r'"key": "run_stop"')


@rfm.simple_test
class CosmoflowGPU(rfm.RegressionTest):
    def __init__(self):
        self.descr = 'CosmoFlow Benchmark'
        self.valid_systems = ["cirrus:compute-gpu"] 
        self.valid_prog_environs = ["*"]

        num_nodes = 16
        self.time_limit = '2h'
        self.num_tasks_per_node = 4
        self.num_tasks = num_nodes*self.num_tasks_per_node
        self.num_cpus_per_task = 8
        self.extra_resources = {'qos': {'qos': 'gpu'}, 
                                "gpu": {"num_gpus_per_node": "4"}}

        self.prerun_cmds = ['eval "$(/work/ta127/ta127/chrisrae/miniconda3/bin/conda shell.bash hook)"', "conda activate mlperf-torch"]
        self.executable = 'python'
        self.executable_opts = ["/work/ta127/ta127/chrisrae/chris-ml-intern/ML_HPC/CosmoFlow/Torch/train.py", "--device", "cuda"]

    @sanity_function
    def assert_target_met():
        return sn.assert_found(r'"key": "target_mae_reached"') and sn.assert_found(r'"key": "run_stop"')