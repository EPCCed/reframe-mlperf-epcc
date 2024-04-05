import yaml
import os
from packaging import version
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
from mlperf_logging import mllog
from mlperf_logging.mllog import constants as log_constants

class SingletonMetaClass(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMetaClass, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

def _run_on_0(func):
    def wrapper(*args, **kwrags):
        if "sync" in kwrags.keys():
            if kwrags["sync"]:
                dist.barrier()
        if dist.get_rank() == 0:
            return func(*args, **kwrags)
    return wrapper

class GlobalContext(dict, metaclass=SingletonMetaClass):
    _config_path = None
    """
    reads the yaml files and stores data as its parameters

    being a singleton class prevents having to read the yaml file every time
    """
    def __init__(self, config_path=None):
        self.mllogger = mllog.get_mllogger()
        if not self.__dict__ and config_path is not None:
            with open(config_path, "r") as stream:
                self.clear()
                self.update(yaml.safe_load(stream))
                if self["device"].lower() == 'gpu':
                    self["device"] = "cuda"
    
    def init_dist(self):
        if dist.is_mpi_available() and not dist.is_torchelastic_launched():
            backend = "mpi"
        elif self.device == "cuda":
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend)
            
    @property
    def rank(self):
        if "rank" not in self.keys():
            self["rank"] = dist.get_rank()
        return self["rank"]
    
    @property
    def world_size(self):
        if "world_size" not in self.keys():
            self["world_size"] = dist.get_world_size()
        return self["world_size"]

    @property
    def local_rank(self):
        if "local_rank" not in self.keys():
            if dist.is_torchelastic_launched():
                self["local_rank"] = int(os.environ['LOCAL_RANK'])
            else:
                # slurm
                taskspernode = int(os.environ["SLURM_NTASKS"]) // int(os.environ["SLURM_NNODES"])
                self["local_rank"] = int(os.environ["SLURM_PROCID"])%taskspernode
        return self["local_rank"]
    
    @property
    def local_world_size(self):
        if "local_world_size" not in self.keys():
            if dist.is_torchelastic_launched():
                self["local_world_size"] = int(os.environ['LOCAL_WORLD_SIZE'])
            else:
                # slurm
                taskspernode = int(os.environ["SLURM_NTASKS"]) // int(os.environ["SLURM_NNODES"])
                self["local_world_size"] = taskspernode
        return self["local_world_size"]
            
    @property
    def device(self):
        return self["device"].lower()
    
    def update_config(self, config_path):
        with open(config_path, "r") as stream:
            self.clear()
            self.update(yaml.safe_load(stream))
            if self["device"].lower() == 'gpu':
                self["device"] = "cuda"
    
    @property
    def gpu_power(self):
        t_ver = version.parse(torch.__version__).release
        if t_ver[0] == 2 and t_ver[1] >= 1 and torch.cuda.is_available() and torch.version.cuda:
            return torch.cuda.power_draw()
        else:
            return 0.0
    
    @property
    def gpu_util(self):
        t_ver = version.parse(torch.__version__).release
        if t_ver[0] == 2 and torch.cuda.is_available() and torch.version.cuda:
            return torch.cuda.utilization()
        else:
            return 0.0
    
    @_run_on_0
    def log_bert(self):
        self.mllogger.default_namespace = "bert"
        self.mllogger.event(key=log_constants.BERT)
        self.mllogger.event(key=log_constants.OPT_NAME, value=self["opt"]["name"])
        self.mllogger.event(key=log_constants.GLOBAL_BATCH_SIZE, value=self["data"]["global_batch_size"])
        self.mllogger.event(key=log_constants.OPT_BASE_LR, value=self["lr_schedule"]["base_lr"])
        self.mllogger.event(key=log_constants.OPT_LAMB_EPSILON, value=1.0e-6)
        self.mllogger.event(key=log_constants.OPT_LR_TRAINING_STEPS, value=self["lr_schedule"]["total_steps"])
        self.mllogger.event(key=log_constants.OPT_LR_WARMUP_STEPS, value=self["lr_schedule"]["lr_warmup_steps"])
        self.mllogger.event(key=log_constants.NUM_WARMUP_STEPS, value=self["lr_schedule"]["lr_warmup_steps"])
        self.mllogger.event(key=log_constants.START_WARMUP_STEP, value=self["lr_schedule"]["start_warmup_step"])
        self.mllogger.event(key=log_constants.OPT_LAMB_BETA_1, value=self["opt"]["betas"][0])
        self.mllogger.event(key=log_constants.OPT_LAMB_BETA_2, value=self["opt"]["betas"][1])
        self.mllogger.event(key=log_constants.OPT_WEIGHT_DECAY, value=self["self"]["weight_decay"])
        self.log_cluster_info()
    
    @_run_on_0
    def log_resnet(self):
        self.mllogger.default_namespace = "resnet"
        self.mllogger.event(key=log_constants.RESNET)
        if self["opt"]["name"].upper() == "SGD":
            self.mllogger.event(key=log_constants.OPT_NAME, value=self["opt"]["name"].upper())
        elif self["opt"]["name"].upper() == "LARS":
            self.mllogger.event(key=log_constants.OPT_NAME, value=self["opt"]["name"].upper())
            self.mllogger.event(key=log_constants.LARS_EPSILON, value=1.0e-6)
        
        self.mllogger.event(key=log_constants.GLOBAL_BATCH_SIZE, value=self["data"]["global_batch_size"])
        self.mllogger.event(key=log_constants.OPT_BASE_LR, value=self["lr_schedule"]["base_lr"])
        self.mllogger.event(key=log_constants.OPT_END_LR, value=self["lr_schedule"]["end_lr"])
        self.mllogger.event(key=log_constants.LARS_OPT_LR_DECAY_POLY_POWER, value=self["lr_schedule"]["poly_power"])
        self.mllogger.event(key=log_constants.OPT_LR_DECAY_STEPS, value=self["lr_schedule"]["decay_steps"])
        self.mllogger.event(key=log_constants.LARS_OPT_MOMENTUM, value=self["opt"]["momentum"])
        self.mllogger.event(key=log_constants.OPT_WEIGHT_DECAY, value=self["opt"]["weight_decay"])
        self.log_cluster_info()

    @_run_on_0
    def log_cluster_info(self):
        if dist.is_torchelastic_launched():
            accels_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
            num_nodes = dist.get_world_size()//accels_per_node
            accels_per_node = accels_per_node if torch.cuda.is_available() else 0
        else:
            num_nodes = int(os.environ["SLURM_NNODES"])
            accels_per_node = dist.get_world_size()//int(os.environ["SLURM_NNODES"]) if torch.cuda.is_available() else 0
        self.mllogger.event(key="number_of_ranks", value=dist.get_world_size())
        self.mllogger.event(key="number_of_nodes", value=num_nodes)
        #accels_per_node = dist.get_world_size()//int(os.environ["SLURM_NNODES"]) if torch.cuda.is_available() else 0
        self.mllogger.event(key="accelerators_per_node", value=accels_per_node)

    @_run_on_0
    def print_0(self, *args, **kwargs):
        print(*args, **kwargs)
    
    """ @contextmanager
    def profiler(self, name: str):
        if self.rank != -1 or not self["training"]["benchmark"]:
            yield None
        else:
            if self.device == "cpu":
                activities=[ProfilerActivity.CPU]
            else:
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            with profile(activities=activities, profile_memory=True, with_flops=True) as prof:
                with record_function(name):
                    yield prof """
    
    @_run_on_0
    def log_event(self, *args, sync=True, **kwargs):
        self.mllogger.event(*args, **kwargs)
    
    @_run_on_0
    def log_seed(self, seed, sync=True):
        self.mllogger.event(key=log_constants.SEED, value=seed)

    @_run_on_0
    def start_init(self, sync=True):
        self.mllogger.start(key=log_constants.INIT_START, value=None)
    
    @_run_on_0
    def stop_init(self, sync=True):
        self.mllogger.end(key=log_constants.INIT_STOP, value=None)
    
    @_run_on_0
    def start_run(self, sync=True):
        self.mllogger.start(key=log_constants.RUN_START, value=None)
    
    @_run_on_0
    def stop_run(self, metadata = {"status": "success"}, sync=True):
        self.mllogger.end(key=log_constants.RUN_STOP, value=None, metadata=metadata)
    
    @_run_on_0
    def start_epoch(self, metadata, sync=True):
        self.mllogger.start(key=log_constants.EPOCH_START, value=None, metadata=metadata)

    @_run_on_0
    def stop_epoch(self, metadata, sync=True):
        self.mllogger.end(key=log_constants.EPOCH_STOP, value=None, metadata=metadata)
    
    @_run_on_0
    def start_eval(self, metadata, sync=True):
        self.mllogger.start(key=log_constants.EVAL_START, value=None, metadata=metadata)
    
    @_run_on_0
    def stop_eval(self, metadata, sync=True):
        self.mllogger.end(key=log_constants.EVAL_STOP, value=None, metadata=metadata)