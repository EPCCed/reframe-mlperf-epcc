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
import poptorch

from ML_HPC.gc import GlobalContext
gc = GlobalContext("/work/ta127/ta127/chrisrae/chris-ml-intern/ML_HPC/CosmoFlow/Torch/config.yaml")
from ML_HPC.CosmoFlow.Torch.model.cosmoflow import StandardCosmoFlow
from ML_HPC.CosmoFlow.Torch.data.TF_record_loader import get_train_dataloader, get_val_dataloader
from ML_HPC.CosmoFlow.Torch.lr_schedule.scheduler import CosmoLRScheduler


#Mean Absolute Error
class MAE:
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
        
        return (self._error / self._items).item()


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

    gc.log_cosmoflow()
    gc.start_init()
    gc.log_seed(1)
    
    train_data = get_train_dataloader(options)
    val_data = get_val_dataloader(val_options)

    net = StandardCosmoFlow()
    
    if gc["opt"]["name"].upper() == "SGD":
        opt = poptorch.optim.SGD(net.parameters(), lr=gc["lr_schedule"]["base_lr"], momentum=gc["opt"]["momentum"], weight_decay=gc["opt"]["weight_decay"])
    else:
        raise NameError(f"Optimiser {gc['opt']['name']} not supported please use SGD")

    model = poptorch.trainingModel(net, options=options, optimizer=opt)

    eval_model = poptorch.inferenceModel(net, options=val_options)

    scheduler = CosmoLRScheduler(opt)
    
    gc.stop_init()
    gc.start_run()

    score = MAE()
    epoch= 0

    while True:
        gc.start_epoch(metadata={"epoch_num": epoch+1})
        for x, y in train_data:
            logits, loss = model(x, y)
        
        gc.log_event(key="learning_rate", value=scheduler.get_last_lr()[0], metadata={"epoch_num": epoch+1})
        gc.log_event(key="train_loss", value=loss.item(), metadata={"epoch_num": epoch+1})

        gc.start_eval(metadata={"epoch_num": epoch+1})
        score.reset()
        avg_eval_loss = 0
        for x, y in val_data:
            logits, loss = eval_model(x, y)
            avg_eval_loss += loss
            score.update(logits, y)
        mae = score.get_value()

        gc.log_event(key="eval_loss", value=(avg_eval_loss/len(val_data)).item(), metadata={"epoch_num": epoch+1})
        gc.log_event(key="eval_mae", value=mae, metadata={"epoch_num": epoch+1})
        gc.stop_eval(metadata={"epoch_num": epoch+1})
        gc.stop_epoch(metadata={"epoch_num": epoch+1})

        
        epoch += 1
        if mae <= gc["training"]["target_mae"] or epoch == gc["data"]["n_epochs"]:
            if mae <= gc["training"]["target_mae"]:
                gc.log_event(key="target_mae_reached", value=gc["training"]["target_mae"], metadata={"epoch_num": epoch+1})
            gc.stop_run()
            break
        
        scheduler.step()


if __name__ == "__main__":
    main()