from pathlib import Path
import sys
import os

#path_root = Path(__file__).parents[3]
path_root = Path(os.getcwd()).parents[2]
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
import ML_HPC.CosmoFlow.Torch.data.TF_record_loader as TF_rl
import ML_HPC.CosmoFlow.Torch.data.h5_dataloader as h5_dl
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
    backend = "mpi:" if dist.get_backend() == "mpi" else "nccl:"
    for event in list(prof.key_averages()):
        if backend in event.key:
            total_time += event.cpu_time_total * 1e-6
            total_time += event.cuda_time_total * 1e-6
    return total_time

@click.command()
@click.option("--device", "-d", default="", show_default=True, type=str, help="The device type to run the benchmark on (cpu|gpu|cuda). If not provided will default to config.yaml")
@click.option("--config", "-c", default=os.path.join(os.getcwd(), "config.yaml"), show_default=True, type=str, help="Path to config.yaml. If not provided will default to config.yaml in the cwd")
@click.option("--data-dir", default=None, show_default=True, type=str, help="Path To DeepCAM dataset. If not provided will deafault to what is provided in the config.yaml")
@click.option("--global_batchsize", "-gbs", default=None, show_default=True, type=int, help="The Global Batchsize")
@click.option("--local_batchsize", "-lbs", default=None, show_default=True, type=int, help="The Local Batchsize, Leave as 0 to use the Global Batchsize")
@click.option("--t_subset_size", default=None, show_default=True, type=int, help="Size of the Training Subset, dont call to use full dataset")
@click.option("--v_subset_size", default=None, show_default=True, type=int, help="Size of the Validation Subset, dont call to use full dataset")
def main(device, config, data_dir, global_batchsize, local_batchsize, t_subset_size, v_subset_size):
    if config:
        gc.update_config(config)
    if device.lower() in ('cpu', "gpu", "cuda"):
        gc["device"] = device.lower()
    if data_dir:
        gc["data"]["data_dir"] = data_dir
    if global_batchsize is not None:
        gc["data"]["global_batch_size"] = global_batchsize
    if local_batchsize is not None:
        gc["data"]["local_batch_size"]= local_batchsize
    if t_subset_size is not None:
        gc["data"]["train_subset"] = t_subset_size
    if v_subset_size is not None:
        gc["data"]["val_subset"] = v_subset_size
        
    torch.backends.cudnn.benchmark = True
    print(gc.device)
    
    torch.manual_seed(1)
    
    gc.init_dist()
    if gc.device == "cuda":
        torch.cuda.set_device("cuda:" + str(gc.local_rank))

    gc.log_cosmoflow()
    gc.start_init()
    gc.log_seed(1)

    if gc["data"]["h5"]:
        get_train_dataloader = h5_dl.get_train_dataloader
        get_val_dataloader = h5_dl.get_val_dataloader
    else:
        get_train_dataloader = TF_rl.get_train_dataloader
        get_val_dataloader = TF_rl.get_val_dataloader
    train_data = get_train_dataloader()
    val_data = get_val_dataloader()
    
    if gc.rank == -1:
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
        total_time = torch.tensor(total_time).to(gc.device)
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
