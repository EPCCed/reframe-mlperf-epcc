import os
from pathlib import Path
import sys
path_root = Path(os.getcwd()).parents[2]
sys.path.append(str(path_root))
import time
import warnings
warnings.filterwarnings("ignore")
import click

import torch
import torch.distributed as dist
import torch.nn as nn
import poptorch
import gcipuinfo

from ML_HPC.gc import GlobalContext
gc = GlobalContext()
import ML_HPC.DeepCAM.Torch.data.data_loader as dl
from ML_HPC.DeepCAM.Torch.model.DeepCAM import DeepLabv3_plus
from ML_HPC.DeepCAM.Torch.lr_scheduler.schedulers import MultiStepLRWarmup, CosineAnnealingLRWarmup
from ML_HPC.DeepCAM.Torch.validation import validate, compute_score
    
def pow_to_float(power):
    try:
        return float(power[:-1])
    except ValueError:
        return 0

@click.command()
@click.option("--device", "-d", default="", show_default="", type=str, help="The device type to run the benchmark on (cpu|gpu|cuda). If not provided will default to config.yaml")
@click.option("--config", "-c", default="", show_default="", type=str, help="Path to config.yaml. If not provided will default to what is provided in train.py")
def main(device, config):
    if config:
        gc.update_config(config)
    if device and device.lower() in ('cpu', "gpu", "cuda"):
        gc["device"] = device.lower()
        
    options = poptorch.Options()
    val_options = poptorch.Options()
    options.deviceIterations(1)
    options.Training.gradientAccumulation(16)
    options.randomSeed(333)
    val_options.replicationFactor(gc["training"]["num_ipus"])
    val_options.deviceIterations(4)
    val_options.randomSeed(333)
    torch.manual_seed(333)

    gc.log_deepcam()
    gc.log_seed(333)
    
    ipu_info = gcipuinfo.gcipuinfo()
    
    gc.start_init()
    
    train_data, train_data_size, val_data, val_data_size = dl.get_dummy_dataloaders(512, options, val_options)  

    net = DeepLabv3_plus(n_input=16, n_classes=3, pretrained=False, rank=0, process_group=None)
    
    if gc["opt"]["name"].upper() == "ADAM":
        opt = poptorch.optim.Adam(net.parameters(), lr=gc["lr_schedule"]["base_lr"], betas=tuple(gc["opt"]["betas"]), eps=1e-6, weight_decay=gc["opt"]["weight_decay"])
    elif gc["opt"]["name"].upper() == "ADAMW":
        opt = poptorch.optim.AdamW(net.parameters(), lr=gc["lr_schedule"]["base_lr"], betas=tuple(gc["opt"]["betas"]), eps=1e-6, weight_decay=gc["opt"]["weight_decay"])
    elif gc["opt"]["name"].upper() == "LAMB":
        opt = poptorch.optim.LAMB(net.parameters(), lr=gc["lr_schedule"]["base_lr"], betas=tuple(gc["opt"]["betas"]), eps=1e-6, weight_decay=gc["opt"]["weight_decay"])
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

    model = poptorch.trainingModel(net, options=options, optimizer=opt)

    eval_model = poptorch.inferenceModel(net, options=val_options)
    
    gc.start_init()
    gc.start_run()

    epoch = 0
    stop_training = False

    print("Training Started \n")
    while True:
        gc.start_epoch(metadata={"epoch_num": epoch+1})
        start = time.time()
        total_power = 0
        for idx, (x, y, _) in enumerate(train_data):
            device_powers = ipu_info.getNamedAttributeForAll(gcipuinfo.IpuPower)
            total_power += sum([pow_to_float(power) for power in device_powers if power != "N/A"])
            logits, loss = model(x, y)
        avg_power = total_power/len(train_data)
        
        total_time = time.time()-start
        total_time = torch.tensor(total_time)
        
        print(f"Time For Epoch: {total_time}")
        print(f"Processing Speed: {4096/total_time}")
        print(f"Avg IPU Usage: {avg_power}")
        

        loss_avg = loss.detach()
        loss_avg_train = loss_avg.item() / float(gc.world_size)

        predictions = torch.argmax(torch.softmax(logits, 1), 1)
        iou = compute_score(predictions, y, num_classes=3)
        iou_avg = iou.detach()
        iou_avg_train = iou_avg.item() / float(gc.world_size)

        gc.log_event(key="learning_rate", value=scheduler.get_last_lr()[0], metadata={"epoch_num": epoch+1})
        gc.log_event(key="training_accuracy", value=iou_avg_train, metadata={"epoch_num": epoch+1})
        gc.log_event(key="train_loss", value=loss_avg_train, metadata={"epoch": epoch+1})


        stop_training = validate(model, val_data, epoch)
        gc.stop_epoch(metadata={"epoch_num": epoch+1})
        epoch += 1

        if stop_training or epoch == gc["data"]["n_epochs"]:
            gc.stop_run()
            break

if __name__ == "__main__":
    main()
    

    