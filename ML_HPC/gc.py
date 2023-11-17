from typing import Any
import yaml

import torch.distributed as dist

from mlperf_logging import mllog
from mlperf_logging.mllog import constants as log_constants

class SingletonMetaClass(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMetaClass, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class GlobalContext(dict, metaclass=SingletonMetaClass):
    _config_path = None
    """
    reads the yaml files and stores data as its parameters

    being a singleton class prevents having to read the yaml file every time
    """
    def __init__(self, config_path=None):
        if not self.__dict__:
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
        return self["device"]
    
    def log_cosmoflow(self):
        mllogger = mllog.get_mllogger()
        mllogger.event(key=log_constants.OPT_NAME, value="SGD")
        mllogger.event(key=log_constants.LARS_OPT_MOMENTUM, value=self["opt"]["momentum"])
        mllogger.event(key=log_constants.OPT_WEIGHT_DECAY, value=self["opt"]["weight_decay"])
        mllogger.event(key="dropout", value=0.5)
        mllogger.event(key=log_constants.OPT_BASE_LR, value=self["lr_schedule"]["base_lr"])
        mllogger.event(key=log_constants.OPT_LR_WARMUP_EPOCHS, value=self["lr_schedule"]["n_warmup_epochs"])
        mllogger.event(key=log_constants.OPT_LR_DECAY_FACTOR, value=max(self["lr_schedule"]["decay_schedule"].values()) if len(self["lr_schedule"]["decay_schedule"])>0 else 1)
    
    def log_deepcam(self):
        mllogger = mllog.get_mllogger()
        mllogger.event(key=log_constants.OPT_NAME, value=self["opt"]["name"].upper())
        if self["opt"]["name"].upper() == "ADAM":
            mllogger.event(key=log_constants.OPT_ADAM_EPSILON, value=1.0e-6)
            mllogger.event(key=log_constants.OPT_ADAM_BETA_1, value=self["opt"]["betas"][0])
            mllogger.event(key=log_constants.OPT_ADAM_BETA_2, value=self["opt"]["betas"][1])
        elif self["opt"]["name"].upper() == "ADAMW":
            mllogger.event(key=log_constants.OPT_ADAMW_EPSILON, value=1.0e-6)
            mllogger.event(key=log_constants.OPT_ADAMW_BETA_1, value=self["opt"]["betas"][0])
            mllogger.event(key=log_constants.OPT_ADAMW_BETA_2, value=self["opt"]["betas"][1])
        elif self["opt"]["name"].upper() == "LAMB":
            mllogger.event(key=log_constants.OPT_LAMB_EPSILON, value=1.0e-6)
            mllogger.event(key=log_constants.OPT_LAMB_BETA_1, value=self["opt"]["betas"][0])
            mllogger.event(key=log_constants.OPT_LAMB_BETA_2, value=self["opt"]["betas"][1])
        
        mllogger.event(key=log_constants.OPT_BASE_LR, value=self["lr_schedule"]["base_lr"])
        mllogger.event(key=log_constants.OPT_LR_WARMUP_STEPS, value=self["lr_schedule"]["lr_warmup_steps"])
        mllogger.event(key=log_constants.OPT_LR_WARMUP_FACTOR, value=self["lr_schedule"]["lr_warmup_factor"])
        mllogger.event(key="scheduler_type", value=self["lr_schedule"]["type"])
        if self["lr_schedule"]["type"].upper() == "MULTISTEP":
            mllogger.event(key="scheduler_milestones", value=self["lr_schedule"]["milestones"])
            mllogger.event(key=log_constants.OPT_LR_DECAY_FACTOR, value=self["lr_schedule"]["decay_rate"])
        if self["lr_schedule"]["type"].upper() == "COSINE_ANNEALING":
            mllogger.event(key="scheduler_t_max", value=self["lr_schedule"]["t_max"])
            mllogger.event(key="scheduler_eta_min", value=self["lr_schedule"]["eta_min"])
        
        mllogger.event(key="gradient_accumulation_frequency", value=1)

    def print_0(self, string):
        if self.rank == 0:
            print(string)

