import os
from pathlib import Path
import sys
path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))
import csv
import click
import time

import torch
import torch.distributed as dist
import poptorch
from torchmetrics.classification import Accuracy
from tqdm import tqdm

from ML.gc import GlobalContext
gc = GlobalContext(
    "/work/ta127/ta127/chrisrae/chris-ml-intern/ML/ResNet50/Torch/config.yaml"
)
import ML.ResNet50.Torch.data.data_loader as dl
from ML.ResNet50.Torch.opt import Lars as LARS
from ML.ResNet50.Torch.model.ResNet import ResNet50

def valid_step(x, y, model, loss_fn, metric_tracker):
    with torch.no_grad():
        logits = model(x)
        loss = loss_fn(logits, y)
        metric_tracker(logits, y)
        return loss


@click.command()
@click.option("--config", "-c", default="", show_default=True, type=str, help="Path to config.yaml. If not provided will default to what is provided in train.py")
def main(config):
    if config:
        gc.update_config(config)

    options = poptorch.Options()
    val_options = poptorch.Options()
    options.replicationFactor(gc["training"]["num_ipus"])
    options.randomSeed(1)
    torch.manual_seed(1)

    train_data = dl.get_train_dataloader(options)
    val_data = dl.get_val_dataloader(val_options)

    net = ResNet50(num_classes=1000)

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

    train_metric = Accuracy(task="multiclass", num_classes=1000)
    val_metric = Accuracy(task="multiclass", num_classes=1000)

    train_metric.to(gc.device)
    val_metric.to(gc.device)

    model.train()

    E = 1
    while True:
        start = time.time()
        for x, y in train_data:
            out, loss = model(x, y)
        
        train_accuracy = train_metric(out, y)
        dist.reduce(train_accuracy, 0)
        total_time = time.time()-start
        total_time = torch.tensor(total_time)
        dist.all_reduce(total_time)
        total_time /= gc.world_size
        if gc.rank == 0:
            print(f"Train Accuracy at Epoch {E}: {train_accuracy/gc.world_size}")
            print(f"Train Loss at Epoch {E}: {loss}")
            dataset_size = gc["data"]["train_subset"] if gc["data"]["train_subset"] else 1000000
            print(f"Processing Speed: {(dataset_size/total_time).item()}")

        if E % 4 == 0:
            for x, y in val_data:
                out, loss = eval_model(x,y)
                val_metric(out, y)
            val_accuracy = val_metric.compute()
            dist.all_reduce(val_accuracy)
            if gc.rank == 0:
                print(f"Train Accuracy at Epoch {E}: {val_accuracy/gc.world_size}")
                print(f"Validation Loss at Epoch {E}: {loss}")
        E += 1
        if "val_accuracy" in dir(): 
            if E == gc["data"]["n_epochs"] or val_accuracy/gc.world_size >= gc["training"]["target_accuracy"]:
                break
        scheduler.step()


if __name__ == "__main__":
    main()
