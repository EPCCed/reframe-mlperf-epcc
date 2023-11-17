import math

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from ML_HPC.gc import GlobalContext
gc = GlobalContext()

class CosmoLRScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1, verbose: bool = False) -> None:
        if gc["lr_schedule"]["scaling"] == "linear":
            scale_factor = gc["data"]["global_batch_size"] / gc["lr_schedule"]["base_batch_size"]
        elif gc["lr_schedule"]["scaling"] == "linear":
            scale_factor = math.sqrt(gc["data"]["global_batch_size"] / gc["lr_schedule"]["base_batch_size"])
        else:
            scale_factor = 1.0
        self.base_lrs = list([i["lr"] for i in optimizer.state_dict()["param_groups"]])
        self.peak_lrs = [base_lr * scale_factor for base_lr in self.base_lrs]
        super().__init__(optimizer, last_epoch, verbose)


    def get_lr(self) -> float:
        if self.last_epoch < gc["lr_schedule"]["n_warmup_epochs"]:
            return [self.last_epoch * (peak_lr - base_lr)/ gc["lr_schedule"]["n_warmup_epochs"] + base_lr for base_lr, peak_lr in zip(self.base_lrs, self.peak_lrs)]
        else:
            decay_factor = 1.
            decay_epoch = 0
            for e, d in gc["lr_schedule"]["decay_schedule"].items():
                if e >= decay_epoch and e < self.last_epoch:
                    decay_epoch, decay_factor = e, d
            return [peak_lr*decay_factor for peak_lr in self.peak_lrs]

            
