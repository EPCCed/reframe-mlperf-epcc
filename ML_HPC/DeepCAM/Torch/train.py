import os
from pathlib import Path
import sys
path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))
import time
from contextlib import contextmanager
import warnings
warnings.filterwarnings("ignore")
import click
from packaging import version

import torch
import torch.distributed as dist
import torch.nn as nn

from ML_HPC.gc import GlobalContext
gc = GlobalContext()
import ML_HPC.DeepCAM.Torch.data.data_loader as dl
from ML_HPC.DeepCAM.Torch.model.DeepCAM import DeepLabv3_plus
from ML_HPC.DeepCAM.Torch.lr_scheduler.schedulers import MultiStepLRWarmup, CosineAnnealingLRWarmup
from ML_HPC.DeepCAM.Torch.optimizer.lamb import Lamb
from ML_HPC.DeepCAM.Torch.validation import validate, compute_score

if version.parse(torch.__version__) < version.parse("2.1.0"):
    get_power = lambda : 0
    print("Torch Version Too Low for GPU Power Metrics")
else:
    get_power = torch.cuda.power_draw


class CELoss(nn.Module):

    def __init__(self, weight):

        # init superclass
        super(CELoss, self).__init__()

        # instantiate loss
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight).to(torch.float32), reduction='none')

    def forward(self, logit, target):

        # squeeze target
        target = target.squeeze(1)
   
        # get losses and predictions
        losses = self.criterion(logit, target)

        # average
        loss = torch.mean(losses)

        return loss

def dummy_loaders(n_samples):
    bs = (gc["data"]["global_batch_size"] // gc.world_size)//gc["data"]["gradient_accumulation"]
    yield torch.ones(bs, 16, 768, 1152),  torch.ones(bs, 1, 768, 1152), "file.txt"

def get_comm_time(prof: torch.profiler.profile):
    total_time = 0
    for event in list(prof.key_averages()):
        if "mpi:" in event.key or "nccl" in event.key:
            total_time += event.cpu_time_total * 1e-6
            total_time += event.cuda_time_total * 1e-6
    return total_time

@click.command()
@click.option("--device", "-d", default="", show_default=True, type=str, help="The device type to run the benchmark on (cpu|gpu|cuda). If not provided will default to config.yaml")
@click.option("--config", "-c", default="", show_default=True, type=str, help="Path to config.yaml. If not provided will default to what is provided in train.py")
@click.option("--data-dir", default=None, show_default=True, type=str, help="Path To DeepCAM dataset. If not provided will deafault to what is provided in the config.yaml")
@click.option("--global-batchsize", "-gbs", default=None, show_default=True, type=int, help="The Global Batchsize")
def main(device, config, data_path, gbs):
    if config:
        gc.update_config(config)
    if device and device.lower() in ('cpu', "gpu", "cuda"):
        gc["device"] = device.lower()
    if data_path:
        gc["data"]["data_dir"] = data_path
    if gbs:
        gc["data"]["global_batch_size"] = gbs


    torch.manual_seed(333)
    gc.init_dist()
    if gc.device == "cuda":
        torch.cuda.set_device("cuda:" + str(gc.local_rank))
    
    gc.start_init()
    
    train_data, train_data_size, val_data, val_data_size = dl.get_dataloaders()  

    model = DeepLabv3_plus(n_input=16, n_classes=3, pretrained=False, rank=gc.rank, process_group=None,).to(gc.device)
    if gc.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)

    if gc["opt"]["name"].upper() == "ADAM":
        opt = torch.optim.Adam(model.parameters(), lr=gc["lr_schedule"]["base_lr"], betas=gc["opt"]["betas"], eps=1e-6, weight_decay=gc["opt"]["weight_decay"])
    elif gc["opt"]["name"].upper() == "ADAMW":
        opt = torch.optim.AdamW(model.parameters(), lr=gc["lr_schedule"]["base_lr"], betas=gc["opt"]["betas"], eps=1e-6, weight_decay=gc["opt"]["weight_decay"])
    elif gc["opt"]["name"].upper() == "LAMB":
        opt = Lamb(model.parameters(), lr=gc["lr_schedule"]["base_lr"], betas=gc["opt"]["betas"], eps=1e-6, weight_decay=gc["opt"]["weight_decay"])
    else:
        raise NameError(f"Optimiser {gc['opt']['name']} not supported please use ADAM|ADAMW|LAMB")
    
    if gc["lr_schedule"]["type"] == "multistep":
        if gc["lr_schedule"]["lr_warmup_steps"] > 0:
            scheduler = MultiStepLRWarmup(opt, last_epoch=-1) # will have to change L_E once checkpointing supported
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, 
                                                             milestones=[gc["lr_schedule"]["milestones"]], 
                                                             gamma=gc["lr_schedule"]["decay_rate"], 
                                                             last_epoch=-1)

    elif gc["lr_schedule"]["type"] == "cosine_annealing":
        if gc["lr_schedule"]["lr_warmup_steps"] > 0:
            scheduler = CosineAnnealingLRWarmup(opt, last_epoch=-1)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                                  T_max=gc["lr_schedule"]["t_max"],
                                                                  eta_min=gc["lr_schedule"]["eta_min"],
                                                                  last_epoch=-1)

    loss_pow = -0.125
    class_weights = [0.986267818390377**loss_pow, 0.0004578708870701058**loss_pow, 0.01327431072255291**loss_pow]
    criterion = CELoss(class_weights).to(gc.device)
    scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=gc["training"]["amp"] and gc.device == "cuda")
    
    gc.stop_init()
    gc.start_run()

    epoch = 0
    stop_training = False
    model.train()
    # Train Loop
    while True:
        gc.start_epoch(metadata={"epoch_num": epoch+1})
        train_data.sampler.set_epoch(epoch)
        model.train()
        start = time.time()
        total_io_time = 0
        with gc.profiler(f"Epoch: {epoch+1}") as prof:
            start_io = time.time_ns()
            for idx, (x, y, _) in enumerate(train_data):
                x, y = x.to(gc.device), y.to(gc.device)
                total_io_time += time.time_ns() - start_io                
                if ((idx + 1)%gc["data"]["gradient_accumulation"]!=0) or (idx+1 != len(train_data)):
                    if isinstance(model, nn.parallel.DistributedDataParallel):
                        with model.no_sync():
                            with torch.autocast(device_type=gc.device, dtype=torch.float16, enabled=gc["training"]["amp"] and gc.device == "cuda"):
                                logits = model.forward(x)
                                loss = criterion.forward(logits, y)/gc["data"]["gradient_accumulation"]
                            scaler.scale(loss).backward()
                        
                    else:
                        with torch.autocast(device_type=gc.device, dtype=torch.float16, enabled=gc["training"]["amp"] and gc.device == "cuda"):
                            logits = model.forward(x)
                            loss = criterion.forward(logits, y)/gc["data"]["gradient_accumulation"]
                        scaler.scale(loss).backward()
                else: 
                    with torch.autocast(device_type=gc.device, dtype=torch.float16, enabled=gc["training"]["amp"] and gc.device == "cuda"):
                        logits = model.forward(x)
                        loss = criterion.forward(logits, y)/gc["data"]["gradient_accumulation"]
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    scheduler.step()
                start_io = time.time_ns()

        total_io_time *= 1e-9
        total_time = time.time()-start
        total_time = torch.tensor(total_time)
        dist.all_reduce(total_time)
        total_time /= gc.world_size
        if gc.rank == 0:
            print(f"Processing Speed: {(train_data_size/total_time).item()}")
            print(f"Time For Epoch: {total_time}")
            print(f"Communication Time: {get_comm_time(prof)}")
            print(f"Total IO Time: {total_io_time}")


        loss_avg = loss.detach()
        if dist.is_initialized():
            dist.reduce(loss_avg, dst=0, op=dist.ReduceOp.SUM)
        loss_avg_train = loss_avg.item() / float(gc.world_size)

        predictions = torch.argmax(torch.softmax(logits, 1), 1)
        iou = compute_score(predictions, y, num_classes=3)
        iou_avg = iou.detach()
        if dist.is_initialized():
            dist.reduce(iou_avg, dst=0, op=dist.ReduceOp.SUM)
        iou_avg_train = iou_avg.item() / float(gc.world_size)

        gc.log_event(key="learning_rate", value=scheduler.get_last_lr()[0], metadata={"epoch_num": epoch+1})
        gc.log_event(key="training_accuracy", value=iou_avg_train, metadata={"epoch_num": epoch+1})
        gc.log_event(key="train_loss", value=loss_avg_train, metadata={"epoch": epoch+1})

        
        stop_training = validate(model, criterion, val_data, epoch)
        gc.stop_epoch(metadata={"epoch_num": epoch+1})
        epoch += 1

        if stop_training or epoch >= gc["data"]["n_epochs"]:
            if stop_training:
                gc.log_event(key="target_iou_met", value=gc["training"]["target_iou"], metadata={"epoch_num": epoch+1})
                gc.stop_run()
            else:
                gc.stop_run(metadata={"status": "target not met"})
            break

if __name__ == "__main__":
    main()
    

    