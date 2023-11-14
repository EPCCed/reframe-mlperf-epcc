import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from flash.core.optimizers import LARS
import torchmetrics

path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))

from ML.gc import GlobalContext

gc = GlobalContext(
    "/work/ta127/ta127/chrisrae/chris-ml-intern/ML/ResNet50/Torch/config.yaml"
)
import ML.ResNet50.Torch.data.data_loader as dl
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

    train_data = dl.get_imagenet_dataloader("train")
    val_data = dl.get_imagenet_dataloader("val")

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

    for E in range(1, gc["data"]["n_epochs"] + 1):
        for i, (x, y) in enumerate(train_data):
            print(x.shape, y.shape)
            x, y = x.to(gc.device), y.to(gc.device)
            loss = train_step(x, y, model, loss_fn, opt, train_metric)
            exit()

        if E % 4 == 0:
            for x, y in val_data:
                loss = valid_step(x, y, model, loss_fn, val_metric)
                print(f"Loss at Epoch {E}: {loss}")
        scheduler.step()


if __name__ == "__main__":
    main()
