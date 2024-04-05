import os
import sys
import inspect
def get_current_file():
    # Get the current frame
    current_frame = inspect.currentframe()
    # Get the outer frame (one level up), to get the caller's file name, not this utility's filename
    caller_frame = inspect.getouterframes(current_frame)[1]
    # Extract the file path
    file_path = caller_frame.filename
    return file_path
path_root = "/".join(get_current_file().split("/")[:-4])
sys.path.append(str(path_root))
import time
import warnings
warnings.filterwarnings("ignore")
import click
import json

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
import nvidia_dlprof_pytorch_nvtx



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

def get_comm_time(prof: torch.profiler.profile):
    total_time = 0
    """ if prof is None:
        return total_time
    backend = "mpi:" if dist.get_backend() == "mpi" else "nccl:"
    for event in list(prof.key_averages()):
        if backend in event.key:
            total_time += event.cpu_time_total * 1e-6
            total_time += event.cuda_time_total * 1e-6 """
    return total_time

@click.command()
@click.option("--device", "-d", default="", show_default=True, type=str, help="The device type to run the benchmark on (cpu|gpu|cuda). If not provided will default to config.yaml")
@click.option("--config", "-c", default=os.path.join(os.getcwd(), "config.yaml"), show_default=True, type=str, help="Path to config.yaml. If not provided will default to config.yaml in the cwd")
@click.option("--data_dir", default=None, show_default=True, type=str, help="Path To DeepCAM dataset. If not provided will deafault to what is provided in the config.yaml")
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

    
    torch.manual_seed(333)
    gc.init_dist()
    if gc.device == "cuda":
        torch.cuda.set_device("cuda:" + str(gc.local_rank))
    

    

    gc.start_init()
    nvidia_dlprof_pytorch_nvtx.init()
    gc.log_seed(333)
    
    train_data, train_data_size, val_data, val_data_size = dl.get_dataloaders()  
    gc.log_deepcam()
    
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
    
    model.eval()
    initial_loss = criterion.forward(model.forward(torch.ones(1, 16, 768, 1152).to(gc.device)), torch.ones(1, 1, 768, 1152, dtype=torch.long).to(gc.device))
    model.train()
    if gc.rank == 0:
        print(json.dumps(dict(gc), indent=2))

    gc.start_run()

    epoch = 0
    stop_training = False

    model.train()
    if torch.cuda.is_available():
        amp_type = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        amp_type = torch.bfloat16
    # Train Loop
    with torch.autograd.profiler.emit_nvtx():
        while True:
            
            gc.start_epoch(metadata={"epoch_num": epoch+1})
            
            model.train()
            start = time.time()
            total_io_time = 0
            power_draw = []
            gpu_utilization = []

            #with gc.profiler(f"Epoch: {epoch+1}") as prof:
            start_io = time.time_ns()
            for idx, (x, y) in enumerate(train_data):
                x, y = x.to(gc.device), y.to(gc.device)

                total_io_time += time.time_ns() - start_io    
                power_draw.append(gc.gpu_power)
                gpu_utilization.append(gc.gpu_util)
                if ((idx + 1)%gc["data"]["gradient_accumulation_freq"]!=0) or (idx+1 != len(train_data)):
                    if isinstance(model, nn.parallel.DistributedDataParallel):
                        with model.no_sync():
                            with torch.autocast(device_type=gc.device, dtype=amp_type, enabled=gc["training"]["amp"] and gc.device == "cuda"):
                                logits = model.forward(x)
                                loss = criterion.forward(logits, y)/gc["data"]["gradient_accumulation_freq"]
                            scaler.scale(loss).backward()
                        
                    else:
                        with torch.autocast(device_type=gc.device, dtype=amp_type, enabled=gc["training"]["amp"] and gc.device == "cuda"):
                            logits = model.forward(x)
                            loss = criterion.forward(logits, y)/gc["data"]["gradient_accumulation_freq"]
                        scaler.scale(loss).backward()
                else: 
                    with torch.autocast(device_type=gc.device, dtype=amp_type, enabled=gc["training"]["amp"] and gc.device == "cuda"):
                        logits = model.forward(x)
                        loss = criterion.forward(logits, y)/gc["data"]["gradient_accumulation_freq"]
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    scheduler.step()
                start_io = time.time_ns()
                    
                if idx % 16 == 0:
                    if gc.rank == 0:
                        print(f"Epoch: {epoch+1} Batch: {idx}/{len(train_data)} Train Time: {time.time()-start} IO Time: {total_io_time*1e-9}")
            
            total_io_time *= 1e-9
            total_time = time.time()-start
            avg_power_draw = torch.mean(torch.tensor(power_draw, dtype=torch.float64)).to(gc.device)
            avg_gpu_util = torch.mean(torch.tensor(gpu_utilization, dtype=torch.float64)).to(gc.device)
            dist.all_reduce(avg_power_draw)
            dist.all_reduce(avg_gpu_util)
            if gc.rank == 0:
                if epoch == 0:
                    print(f"Change In Train Loss at Epoch: {initial_loss - loss}")
                print(f"Change In Train Loss at Epoch {epoch}: {loss}")
                print(f"Processing Speed: {(train_data_size/total_time)}")
                print(f"Time For Epoch: {total_time}")
                #print(f"Communication Time: {get_comm_time(prof)}")
                if gc.device == "cuda":
                    print(f"Avg GPU Power Draw: {avg_power_draw*1e-3:.5f}")
                    print(f"Avg GPU Utilization: {avg_gpu_util/gc.world_size:.2f}")
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
    

    
