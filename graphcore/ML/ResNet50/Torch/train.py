import os
from pathlib import Path
import sys
path_root = Path(os.getcwd()).parents[2]
sys.path.append(str(path_root))
import csv
import click
import time

import torch
import poptorch
import gcipuinfo
from torchmetrics.classification import Accuracy
from tqdm import tqdm

from ML.gc import GlobalContext
gc = GlobalContext()
import ML.ResNet50.Torch.data.data_loader as dl
from ML.ResNet50.Torch.opt import Lars as LARS
from ML.ResNet50.Torch.model.ResNet import ResNet50

def valid_step(x, y, model, loss_fn, metric_tracker):
    with torch.no_grad():
        logits = model(x)
        loss = loss_fn(logits, y)
        metric_tracker(logits, y)
        return loss

def pow_to_float(power):
    try:
        return float(power[:-1])
    except ValueError:
        return 0
    
def get_ipu_power_all():
    device_powers = ipu_info.getNamedAttributeForAll(gcipuinfo.IpuPower)
    return [pow_to_float(pow) for pow in device_powers if pow != "N/A"]


@click.command()
@click.option("--config", "-c", default="", show_default=True, type=str, help="Path to config.yaml. If not provided will default to what is provided in train.py")
def main(config):
    if config:
        gc.update_config(config)

    options = poptorch.Options()
    val_options = poptorch.Options()
    options.Precision.setPartialsType(torch.float16)
    options.replicationFactor(4)
    options.deviceIterations(32)
    options.setExecutionStrategy(poptorch.ShardedExecution())
    options.Training.gradientAccumulation(8)
    val_options.replicationFactor(gc["training"]["num_ipus"])
    
    ipu_info = gcipuinfo.gcipuinfo()

    options.randomSeed(1)
    torch.manual_seed(1)

    train_data = dl.get_train_dataloader(options)
    val_data = dl.get_val_dataloader(val_options)

    net = ResNet50(num_classes=1000)
    net.train()

    if gc["opt"]["name"].upper() == "SGD":
        opt = torch.optim.SGD(
            net.parameters(),
            lr=gc["lr_schedule"]["base_lr"],
            momentum=gc["opt"]["momentum"],
            weight_decay=gc["opt"]["weight_decay"],
        )
    elif gc["opt"]["name"].upper() == "LARS":
        raise AttributeError("LARS optimizer is not supported on graphcore")
    else:
        raise ValueError(
            f"Optimiser {gc['opt']['name']} not supported please use SGD|LARS"
        )

    model = poptorch.trainingModel(net, options=options, optimizer=opt)

    eval_model = poptorch.inferenceModel(net, options=val_options)

    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        opt, total_iters=gc["data"]["n_epochs"], power=gc["lr_schedule"]["poly_power"]
    )

    model.train()

    E = 1
    while True:
        start = time.time()
        total_power = 0
        for x, y in train_data:
            device_powers = ipu_info.getNamedAttributeForAll(gcipuinfo.IpuPower)
            total_power += sum([pow_to_float(power) for power in device_powers if power != "N/A"])
            loss, out = model(x, y)
        avg_power = total_power/len(train_data)
        
        #train_accuracy = train_metric(out, y)
        total_time = time.time()-start
        total_time = torch.tensor(total_time)
        #print(f"Train Accuracy at Epoch {E}: {train_accuracy}")
        dataset_size = gc["data"]["train_subset"] if gc["data"]["train_subset"] else 1281167
        print(f"Processing Speed: {(dataset_size/total_time).item()}")
        print(f"Time For Epoch: {total_time}")
        print(f"Train Loss at Epoch {E}: {loss}")
        print(f"Avg IPU Power Usage: {avg_power}")
        print("\n")

        if E % 4 == 0:
            val_accuracy = 0
            #for x, y in val_data:
                #out, loss = eval_model(x,y)
                #val_metric(out, y)
            #val_accuracy = val_metric.compute()
            #print(f"Train Accuracy at Epoch {E}: {val_accuracy}")
            #print(f"Validation Loss at Epoch {E}: {loss}"
        E += 1
        if "val_accuracy" in dir(): 
            if E == gc["data"]["n_epochs"] or val_accuracy >= gc["training"]["target_accuracy"]:
                break
        #scheduler.step()


if __name__ == "__main__":
    main()
