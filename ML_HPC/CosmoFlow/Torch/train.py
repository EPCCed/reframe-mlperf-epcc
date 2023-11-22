from pathlib import Path
import sys
import os
path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))
import time
import warnings
warnings.filterwarnings("ignore")
import click

import torch 
import torch.nn as nn
import torch.distributed as dist

from ML_HPC.gc import GlobalContext
gc = GlobalContext("/work/ta127/ta127/chrisrae/chris-ml-intern/ML_HPC/CosmoFlow/Torch/config.yaml")
from ML_HPC.CosmoFlow.Torch.model.cosmoflow import StandardCosmoFlow
from ML_HPC.CosmoFlow.Torch.data.TF_record_loader import get_train_dataloader, get_val_dataloader
from ML_HPC.CosmoFlow.Torch.lr_schedule.scheduler import CosmoLRScheduler


#Mean Absolute Error
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

@click.command()
@click.option("--device", "-d", default="", show_default=True, type=str, help="The device type to run the benchmark on (cpu|gpu|cuda). If not provided will default to config.yaml")
@click.option("--config", "-c", default="", show_default=True, type=str, help="Path to config.yaml. If not provided will default to what is provided in train.py")
def main(device, config):
    if device and device.lower() in ('cpu', "gpu", "cuda"):
        gc["device"] = device.lower()
    if config:
        gc.update_config(config)

    torch.manual_seed(1)
    if dist.is_mpi_available():
        backend = "mpi"
    elif gc.device == "cuda":
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend)

    gc.log_cosmoflow()
    gc.start_init()
    gc.log_seed(1)

    if gc.device == "cuda":
        local_rank = os.environ["LOCAL_RANK"]
        torch.cuda.set_device("cuda:" + local_rank)
    
    train_data = get_train_dataloader()
    val_data = get_val_dataloader()

    model = StandardCosmoFlow().to(gc.device)
    if gc.world_size > 1:
        model = nn.parallel.DistributedDataParallel(model)
    
    if gc["opt"]["name"].upper() == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=gc["lr_schedule"]["base_lr"], momentum=gc["opt"]["momentum"], weight_decay=gc["opt"]["weight_decay"])
    else:
        raise NameError(f"Optimiser {gc['opt']['name']} not supported please use SGD")

    scheduler = CosmoLRScheduler(opt)

    criterion = nn.MSELoss()
    
    gc.stop_init()
    gc.start_run()

    score = DistributedMAE()
    epoch= 0

    while True:
        model.train()
        with gc.profiler(f"Epoch: {epoch+1}") as prof:
            gc.start_epoch(metadata={"epoch_num": epoch+1})
            for x, y in train_data:
                opt.zero_grad()

                x, y = x.to(gc.device), y.to(gc.device)

                logits = model.forward(x)
                loss = criterion.forward(logits, y)
                loss.backward()

                opt.step()
            
            gc.log_event(key="learning_rate", value=scheduler.get_last_lr()[0], metadata={"epoch_num": epoch+1})
            gc.log_event(key="train_loss", value=loss.item(), metadata={"epoch_num": epoch+1})

            gc.start_eval(metadata={"epoch_num": epoch+1})
            model.eval()
            score.reset()
            avg_eval_loss = 0
            with torch.no_grad():
                for x, y in val_data:
                    x, y = x.to(gc.device), y.to(gc.device)
                    logits = model.forward(x)
                    avg_eval_loss += criterion.forward(logits, y)
                    score.update(logits, y)
                mae = score.get_value()

                gc.log_event(key="eval_loss", value=(avg_eval_loss/len(val_data)).item(), metadata={"epoch_num": epoch+1})
                gc.log_event(key="eval_mae", value=mae, metadata={"epoch_num": epoch+1})
            gc.stop_eval(metadata={"epoch_num": epoch+1})
            gc.stop_epoch(metadata={"epoch_num": epoch+1})
        if gc.rank == 0:
            print(prof.key_averages().table(sort_by="cpu_time_total"))
        
        epoch += 1
        if mae <= gc["training"]["target_mae"] or epoch == gc["data"]["n_epochs"]:
            if mae <= gc["training"]["target_mae"]:
                gc.log_event(key="target_mae_reached", value=gc["training"]["target_mae"], metadata={"epoch_num": epoch+1})
            gc.stop_run()
            break
        
        scheduler.step()


if __name__ == "__main__":
    main()