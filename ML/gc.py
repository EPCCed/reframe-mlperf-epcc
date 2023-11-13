import yaml
import torch.distributed as dist

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