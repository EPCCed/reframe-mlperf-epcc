from pathlib import Path
import sys
import os
path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))

import torch 
import torch.nn as nn
import torch.distributed as dist

from ML_HPC.gc import GlobalContext
gc = GlobalContext("/work/ta127/ta127/chrisrae/chris-ml-intern/ML_HPC/CosmoFlow/Torch/config.yaml")
from ML_HPC.CosmoFlow.Torch.model.cosmoflow import StandardCosmoFlow
from ML_HPC.CosmoFlow.Torch.data.CPU_data_loader import CosmoData
from ML_HPC.CosmoFlow.Torch.lr_schedule.scheduler import CosmoLRScheduler

class DistributedMAE:
    def __init__(self):
        self.reset()

        self.op = torch.nn.L1Loss(reduction="sum")
    
    def reset(self):
        self._items = 0
        self._error = 0.0
    
    def update(self, logits: torch.Tensor, label: torch.Tensor):
        self._error += self.op(logits, label)
        self._items += logits.numel()

    def get_value(self):
        if self._items == 0:
            return 0
        
        elif gc.world_size == 1:
            return (self._error / self._items).item()
        else:
            info_tensor = torch.tensor([self._error, self._items])
            dist.all_reduce(info_tensor)
            return (info_tensor[0]/ info_tensor[1]).item()

def main():
    torch.manual_seed(0)
    if dist.is_mpi_available():
        backend = "mpi"
    elif gc.device == "cuda":
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend)

    if gc.device == "cuda":
        local_rank = os.environ["LOCAL_RANK"]
        torch.cuda.set_device("cuda:" + local_rank)
    
    train_data, val_data = ...

    model = StandardCosmoFlow().to(gc.device)
    if gc.world_size > 1:
        model = nn.parallel.DistributedDataParallel(model)
    
    if gc["opt"]["name"].upper() == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=gc["lr_schedule"]["base_lr"], momentum=gc["opt"]["momentum"])
    else:
        raise NameError(f"Optimiser {gc['opt']['name']} not supported please use SGD")

    scheduler = CosmoLRScheduler(opt)

    criterion = nn.MSELoss()

    score = DistributedMAE()
    epoch= 0

    while True:
        model.train()
        for x, y in train_data:
            opt.zero_grad()

            x, y = x.to(gc.device), y.to(gc.device)

            logits = model.forward(x)
            loss = criterion.forward(logits, y)
            loss.backward()

            opt.step()
        
        model.eval()
        score.reset()
        with torch.no_grad():
            for x, y in val_data:
                x, y = x.to(gc.device), y.to(gc.device)
                logits = model.forward(x)
                score.update(logits, y)
            mae = score.get_value()
        
        epoch += 1
        if mae >= gc["training"]["target_mae"] or epoch == gc["data"]["n_epochs"]:
            break
        
        scheduler.step()


if __name__ == "__main__":
    main()
