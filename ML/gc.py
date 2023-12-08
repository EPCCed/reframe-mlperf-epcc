import yaml
import os
import time
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
        if not self.__dict__ and config_path is not None:
            with open(config_path, "r") as stream:
                self.clear()
                self.update(yaml.safe_load(stream))
                if self["device"].lower() == 'gpu':
                    self["device"] = "cuda"
            
    @property
    def rank(self):
        return dist.get_rank()
    
    @property
    def world_size(self):
        return dist.get_world_size()
    
    @property
    def device(self):
        return self["device"].lower()
    
    def update_config(self, config_path):
        with open(config_path, "r") as stream:
            self.__dict__.clear()
            self.__dict__.update(yaml.safe_load(stream))
            if self["device"].lower() == 'gpu':
                self["device"] = "cuda"
    
    @property
    def mpi_log_hook(self):
        # probably wont work due to asynchronous property of futures
        if not hasattr(self, "_mpi_total_time"):
            self._mpi_total_time = 0
        if not hasattr(self, "_mpi_iter_start_time"):
            self._mpi_iter_start_time = 0

        def _call_start_timer():
            self._mpi_iter_start_time = time.time_ns()

        def _call_log_timer():
            self._mpi_total_time += time.time_ns() - self._mpi_iter_start_time

        def _time_mpi(state, bucket):
            _call_start_timer()
            fut = dist.all_reduce(bucket.buffer()).get_future()
            _call_log_timer()
            return fut
        return _time_mpi

    def get_mpi(self):
        return self._mpi_total_time*1e-9
    
    @_run_on_0
    def log_bert(self):
        self.mllogger = mllog.get_mllogger()
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
        self.mllogger = mllog.get_mllogger()
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
        self.mllogger.event(key="number_of_ranks", value=dist.get_world_size())
        self.mllogger.event(key="number_of_nodes", value=int(os.environ["SLURM_NNODES"]))
        accels_per_node = dist.get_world_size()//int(os.environ["SLURM_NNODES"]) if torch.cuda.is_available() else 0
        self.mllogger.event(key="accelerators_per_node", value=accels_per_node)

    @_run_on_0
    def print_0(self, *args, **kwargs):
        print(*args, **kwargs)
    
    @contextmanager
    def profiler(self, name: str):
        if self.rank != 0:
            yield None
        else:
            if self.device == "cpu":
                activities=[ProfilerActivity.CPU]
            else:
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            with profile(activities=activities, with_flops=True) as prof:
                with record_function(name):
                    yield prof
    
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