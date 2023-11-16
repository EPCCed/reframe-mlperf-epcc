import yaml
import torch.distributed as dist

from mlperf_logging import mllog
from mlperf_logging.mllog import constants as log_constants

class SingletonMetaClass(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMetaClass, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]


class GlobalContext(dict, metaclass=SingletonMetaClass):
    """
    reads the yaml files and stores data as its parameters

    being a singlton class prevents having to read the yaml file every time
    """
    def __init__(self, config_path=None):
        super().__init__()
        self.__dict__ = self
        if not self.__dict__:
            with open(config_path, 'r') as stream:
                config = yaml.safe_load(stream)
            for k, v in config.items():
                self[k] = v
        
        if self["device"] == "gpu":
            self["device"] = "cuda"

            
    @property
    def rank(self):
        return dist.get_rank()
    
    @property
    def world_size(self):
        return dist.get_world_size()
    
    def log_bert(self):
        mllogger = mllog.get_mllogger()
        mllogger.event(key=log_constants.BERT)
        mllogger.event(key=log_constants.OPT_NAME, value=self["opt"]["name"])
        mllogger.event(key=log_constants.GLOBAL_BATCH_SIZE, value=self["data"]["global_batch_size"])
        mllogger.event(key=log_constants.OPT_BASE_LR, value=self["lr_schedule"]["base_lr"])
        mllogger.event(key=log_constants.OPT_LAMB_EPSILON, value=1.0e-6)
        mllogger.event(key=log_constants.OPT_LR_TRAINING_STEPS, value=self["lr_schedule"]["total_steps"])
        mllogger.event(key=log_constants.OPT_LR_WARMUP_STEPS, value=self["lr_schedule"]["lr_warmup_steps"])
        mllogger.event(key=log_constants.NUM_WARMUP_STEPS, value=self["lr_schedule"]["lr_warmup_steps"])
        mllogger.event(key=log_constants.START_WARMUP_STEP, value=self["lr_schedule"]["start_warmup_step"])
        mllogger.event(key=log_constants.OPT_LAMB_BETA_1, value=self["opt"]["betas"][0])
        mllogger.event(key=log_constants.OPT_LAMB_BETA_2, value=self["opt"]["betas"][1])
        mllogger.event(key=log_constants.OPT_WEIGHT_DECAY, value=self["self"]["weight_decay"])
    
    def log_resnet(self):
        mllogger = mllog.get_mllogger()
        mllogger.event(key=log_constants.RESNET)
        if self["opt"]["name"].upper() == "SGD":
            mllogger.event(key=log_constants.OPT_NAME, value=self["opt"]["name"].upper())
        elif self["opt"]["name"].upper() == "LARS":
            mllogger.event(key=log_constants.OPT_NAME, value=self["opt"]["name"].upper())
            mllogger.event(key=log_constants.LARS_EPSILON, value=1.0e-6)
        
        mllogger.event(key=log_constants.GLOBAL_BATCH_SIZE, value=self["data"]["global_batch_size"])
        mllogger.event(key=log_constants.OPT_BASE_LR, value=self["lr_schedule"]["base_lr"])
        mllogger.event(key=log_constants.OPT_END_LR, value=self["lr_schedule"]["end_lr"])
        mllogger.event(key=log_constants.LARS_OPT_LR_DECAY_POLY_POWER, value=self["lr_schedule"]["poly_power"])
        mllogger.event(key=log_constants.OPT_LR_DECAY_STEPS, value=self["lr_schedule"]["decay_steps"])
        mllogger.event(key=log_constants.LARS_OPT_MOMENTUM, value=self["opt"]["momentum"])
        mllogger.event(key=log_constants.OPT_WEIGHT_DECAY, value=self["opt"]["weight_decay"])
