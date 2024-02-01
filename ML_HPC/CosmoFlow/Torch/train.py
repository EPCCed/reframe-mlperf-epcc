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
from tqdm import tqdm

from ML_HPC.gc import GlobalContext
gc = GlobalContext()
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

def get_comm_time(prof: torch.profiler.profile):
    total_time = 0
    for event in list(prof.key_averages()):
        if "mpi:" in event.key:
            total_time += event.cpu_time_total * 1e-6
            total_time += event.cuda_time_total * 1e-6
    return total_time

@click.command()
@click.option("--device", "-d", show_default=True, type=str, help="The device type to run the benchmark on (cpu|gpu|cuda). If not provided will default to config.yaml")
@click.option("--config", "-c", show_default=True, type=str, help="Path to config.yaml. If not provided will default to what is provided in train.py")
@click.option("--data-dir", default=None, show_default=True, type=str, help="Path To DeepCAM dataset. If not provided will deafault to what is provided in the config.yaml")
@click.option("--global-batchsize", "-gbs", default=None, show_default=True, type=int, help="The Global Batchsize")
@click.option("--local-batchsize", "-lbs", default=0, show_default=True, type=int, help="The Local Batchsize, Leave as 0 to use the Global Batchsize")
@click.option("--t_subset_size", default=0, show_default=True, type=int, help="Size of the Training Subset, dont call to use full dataset")
@click.option("--v_subset_size", default=0, show_default=True, type=int, help="Size of the Validation Subset, dont call to use full dataset")
def main(device, config, data_path, gbs, lbs, t_subset, v_subset):
    if config:
        gc.update_config(config)
    if device and device.lower() in ('cpu', "gpu", "cuda"):
        gc["device"] = device.lower()
    if data_path:
        gc["data"]["data_dir"] = data_path
    if gbs:
        gc["data"]["global_batch_size"] = gbs
    if lbs:
        gc["data"]["local_batch_size"]= lbs
    if t_subset:
        gc["data"]["n_train"] = t_subset
    if v_subset:
        gc["data"]["n_eval"] = v_subset
        
    torch.backends.cudnn.benchmark = True
    print(gc.device)
    
    torch.manual_seed(1)
    
    gc.init_dist()
    if gc.device == "cuda":
        torch.cuda.set_device("cuda:" + str(gc.local_rank))

    gc.log_cosmoflow()
    gc.start_init()
    gc.log_seed(1)

    
    train_data = get_train_dataloader()
    val_data = get_val_dataloader()
    
    if gc.rank == 0:
        train_data = tqdm(train_data, unit="inputs", unit_scale=(gc["data"]["global_batch_size"] // gc.world_size)//gc["data"]["gradient_accumulation_freq"])

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
        gc.start_epoch(metadata={"epoch_num": epoch+1})
        start = time.time()
        total_io_time = 0
        with gc.profiler(f"Epoch: {epoch+1}") as prof:
            start_io = time.time_ns()
            for b_idx, (x, y) in enumerate(train_data):
                total_io_time += time.time_ns() - start_io
                x, y = x.to(gc.device), y.to(gc.device)
                
                if b_idx%gc["data"]["gradient_accumulation_freq"] != 0:
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        with model.no_sync():
                            logits = model.forward(x)
                            loss = criterion.forward(logits, y)/gc["data"]["gradient_accumulation_freq"]
                            loss.backward()
                    else:
                        logits = model.forward(x)
                        loss = criterion.forward(logits, y)/gc["data"]["gradient_accumulation_freq"]
                        loss.backward()      
                else:
                    logits = model.forward(x)
                    loss = criterion.forward(logits, y)/gc["data"]["gradient_accumulation_freq"]
                    loss.backward()

                    opt.step()
                    opt.zero_grad()
            
                start_io = time.time_ns()
        
        total_io_time *= 1e-9
        
        total_time = time.time()-start
        total_time = torch.tensor(total_time)
        dist.all_reduce(total_time)
        total_time /= gc.world_size
        if gc.rank == 0:
            print(f"Train Loss at Epoch {epoch+1}: {loss}")
            dataset_size = gc["data"]["n_train"]
            print(f"Processing Speed: {(dataset_size/total_time).item()}")
            print(f"Time For Epoch: {total_time}")
            print(f"Communication Time: {get_comm_time(prof)}")
            print(f"Total IO Time: {total_io_time}")

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
        
        epoch += 1
        if mae <= gc["training"]["target_mae"] or epoch >= gc["data"]["n_epochs"]:
            if mae <= gc["training"]["target_mae"]:
                gc.log_event(key="target_mae_reached", value=gc["training"]["target_mae"], metadata={"epoch_num": epoch+1})
            gc.stop_run()
            break
        
        scheduler.step()


if __name__ == "__main__":
    main()