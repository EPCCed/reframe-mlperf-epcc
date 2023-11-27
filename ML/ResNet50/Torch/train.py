import os
from pathlib import Path
import sys
path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))
import click

import torch
import torch.distributed as dist
import torchmetrics

from ML.gc import GlobalContext
gc = GlobalContext(
    "/work/z043/z043/crae/chris-ml-intern/ML/ResNet50/Torch/config.yaml"
)
import ML.ResNet50.Torch.data.data_loader as dl
from ML.ResNet50.Torch.opt import Lars as LARS
from ML.ResNet50.Torch.model.ResNet import ResNet50


def train_step(x, y, model, loss_fn, opt, metric_tracker):
    opt.zero_grad()
    logits = model(x)
    loss = loss_fn(logits, y)
    metric_tracker(logits, y)
    loss.backward()
    opt.step()
    return loss


def valid_step(x, y, model, loss_fn, metric_tracker):
    with torch.no_grad():
        logits = model(x)
        loss = loss_fn(logits, y)
        metric_tracker(logits, y)
        return loss


@click.command()
@click.option("--device", "-d", default="", show_default=True, type=str, help="The device type to run the benchmark on (cpu|gpu|cuda). If not provided will default to config.yaml")
@click.option("--config", "-c", default="", show_default=True, type=str, help="Path to config.yaml. If not provided will default to what is provided in train.py")
def main(device, config):
    if device and device.lower() in ('cpu', "gpu", "cuda"):
        gc["device"] = device.lower()
    if config:
        gc.update_config(config)
    
    torch.manual_seed(0)
    if dist.is_mpi_available():
        backend = "mpi"
    elif gc.device == "cuda":
        backend = "nccl"
    else:
        backend = "gloo"
    print(backend)
    dist.init_process_group(backend)

    if gc.device == "cuda":
        local_rank = os.environ["LOCAL_RANK"]
        torch.cuda.set_device("cuda:" + local_rank)

    train_data = dl.get_train_dataloader()
    val_data = dl.get_val_dataloader()

    model = ResNet50(num_classes=1000).to(gc.device)
    if gc.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)
        if gc.device == "cpu":
            pass
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if gc["opt"]["name"].upper() == "SGD":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=gc["lr_schedule"]["base_lr"],
            momentum=gc["opt"]["momentum"],
            weight_decay=gc["opt"]["weight_decay"],
        )
    elif gc.opt.name.upper() == "LARS":
        opt = LARS(
            model.parameters(),
            lr=gc["lr_schedule"]["base_lr"],
            momentum=gc["opt"]["momentum"],
            weight_decay=gc["opt"]["weight_decay"],
        )
    else:
        raise ValueError(
            f"Optimiser {gc['opt']['name']} not supported please use SGD|LARS"
        )

    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        opt, total_iters=gc["data"]["n_epochs"], power=gc["lr_schedule"]["poly_power"]
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    train_metric = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=1000
    )
    val_metric = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=1000
    )

    train_metric.to(gc.device)
    val_metric.to(gc.device)

    model.train()

    for E in range(1, gc["data"]["n_epochs"]+1):
        for i, (x, y) in enumerate(train_data):
            print(x.shape, y.shape)
            x, y = x.to(gc.device), y.to(gc.device)
            loss = train_step(x, y, model, loss_fn, opt, train_metric)

        if E % 4 == 0:
            for x, y in val_data:
                loss = valid_step(x, y, model, loss_fn, val_metric)
                print(f"Loss at Epoch {E}: {loss}")
        scheduler.step()


if __name__ == "__main__":
    main()
