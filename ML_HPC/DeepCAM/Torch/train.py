from pathlib import Path
import sys
path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))

import os
from collections import Counter
import math
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from flash.core.optimizers import LAMB

from ML_HPC.gc import GlobalContext
gc = GlobalContext("/work/ta127/ta127/chrisrae/chris-ml-intern/ML_HPC/DeepCAM/Torch/config.yaml")
import ML_HPC.DeepCAM.Torch.data.CPU_data_loader as dl
from ML_HPC.DeepCAM.Torch.model.DeepCAM import DeepLabv3_plus
from ML_HPC.DeepCAM.Torch.validation import validate, compute_score


class CELoss(nn.Module):

    def __init__(self, weight):

        # init superclass
        super(CELoss, self).__init__()

        # instantiate loss
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight).float(), reduction='none')


    def forward(self, logit, target):

        # squeeze target
        target = target.squeeze(1)
        
        # get losses and predictions
        losses = self.criterion(logit, target)

        # average
        loss = torch.mean(losses)

        return loss


class MultiStepLRWarmup(_LRScheduler):

    @torch.jit.ignore
    def __init__(self, optimizer, warmup_steps, warmup_factor,
                 milestones, gamma=0.1, last_epoch=-1, verbose=False):
        
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.warmup_slope = 1./float(self.warmup_steps) if self.warmup_steps > 0 else 1.
        self.milestones = Counter([x + self.warmup_steps + 1 for x in milestones])
        self.gamma = gamma
        super(MultiStepLRWarmup, self).__init__(optimizer, last_epoch, verbose)

    @torch.jit.export
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        # compute LR
        if self.last_epoch >= self.warmup_steps:
            # decay phase
            if self.last_epoch not in self.milestones:
                return [group['lr'] for group in self.optimizer.param_groups]
            return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                    for group in self.optimizer.param_groups] 

        else:
            # linear warmup phase
            if self.warmup_factor == 1.0:
                return [base_lr * (float(self.last_epoch) * self.warmup_slope)
                        for base_lr in self.base_lrs]
            else:
                return [base_lr * ((self.warmup_factor - 1.) * float(self.last_epoch) * self.warmup_slope + 1.)
                        for base_lr in self.base_lrs]


class CosineAnnealingLRWarmup(_LRScheduler):

    @torch.jit.ignore
    def __init__(self, optimizer, warmup_steps, warmup_factor,
                 T_max, eta_min=0, last_epoch=-1, verbose=False):
        
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.warmup_slope = 1./float(self.warmup_steps) if self.warmup_steps > 0 else 1.
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLRWarmup, self).__init__(optimizer, last_epoch, verbose)

    @torch.jit.export     
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        # compute LR
        if self.last_epoch >= self.warmup_steps:
            # cosine phase
            last_epoch_eff = self.last_epoch - self.warmup_steps
            if last_epoch_eff == 0:
                return [group['lr'] for group in self.optimizer.param_groups]
            elif (last_epoch_eff - 1 - self.T_max) % (2 * self.T_max) == 0:
                return [group['lr'] + (base_lr - self.eta_min) *
                        (1 - math.cos(math.pi / self.T_max)) / 2
                        for base_lr, group in
                        zip(self.base_lrs, self.optimizer.param_groups)]
            return [(1 + math.cos(math.pi * last_epoch_eff / self.T_max)) /
                    (1 + math.cos(math.pi * (last_epoch_eff - 1) / self.T_max)) *
                    (group['lr'] - self.eta_min) + self.eta_min
                    for group in self.optimizer.param_groups]

        else:
            # linear warmup phase
            if self.warmup_factor == 1.0:
                return [base_lr * (float(self.last_epoch) * self.warmup_slope)
                        for base_lr in self.base_lrs]
            else:
                return [base_lr * ((self.warmup_factor - 1.) * float(self.last_epoch) * self.warmup_slope + 1.)
                        for base_lr in self.base_lrs]


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
    
    train_data, train_data_size, val_data, val_data_size = dl.get_dataloaders()

    model = DeepLabv3_plus(n_input=16, n_classes=3, pretrained=False, rank=gc.rank, process_group=None).to(gc.device)
    if gc.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)

    if gc["opt"]["name"].upper() == "ADAM":
        opt = torch.optim.Adam(model.parameters(), lr=gc["lr_schedule"]["base_lr"], betas=gc["opt"]["betas"], eps=1e-6, weight_decay=gc["opt"]["weight_decay"])
    elif gc["opt"]["name"].upper() == "ADAMW":
        opt = torch.optim.AdamW(model.parameters(), lr=gc["lr_schedule"]["base_lr"], betas=gc["opt"]["betas"], eps=1e-6, weight_decay=gc["opt"]["weight_decay"])
    elif gc["opt"]["name"].upper() == "LAMB":
        opt = LAMB(model.parameters(), lr=gc["lr_schedule"]["base_lr"], betas=gc["opt"]["betas"], eps=1e-6, weight_decay=gc["opt"]["weight_decay"])
    else:
        raise NameError(f"Optimiser {gc['opt']['name']} not supported please use ADAM|ADAMW|LAMB")
    
    if gc["lr_schedule"]["type"] == "multistep":
        milestones = [gc["lr_schedule"]["milestones"]]
        gamma = gc["lr_schedule"]["decay_rate"]

        if gc["lr_schedule"]["lr_warmup_steps"] > 0:
            scheduler = MultiStepLRWarmup(opt, 
                                          warmup_steps=gc["lr_schedule"]["lr_warmup_steps"], 
                                          warmup_factor=gc["lr_schedule"]["lr_warmup_factor"], 
                                          milestones=milestones, 
                                          gamma=gamma, 
                                          last_epoch=-1) # will have to change once checkpointing supported
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, 
                                                             milestones=milestones, 
                                                             gamma=gamma, 
                                                             last_epoch=-1)
    elif gc["lr_schedule"]["type"] == "cosine_annealing":
        if gc["lr_schedule"]["lr_warmup_steps"] > 0:
            scheduler = CosineAnnealingLRWarmup(opt, 
                                                warmup_steps=gc["lr_schedule"]["lr_warmup_steps"], 
                                                warmup_factor=gc["lr_schedule"]["lr_warmup_factor"],
                                                T_max=gc["lr_schedule"]["t_max"],
                                                eta_min=gc["lr_schedule"]["eta_min"],
                                                last_epoch=-1)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                                  T_max=gc["lr_schedule"]["t_max"],
                                                                  eta_min=gc["lr_schedule"]["eta_min"],
                                                                  last_epoch=-1)

    loss_pow = -0.125
    class_weights = [0.986267818390377**loss_pow, 0.0004578708870701058**loss_pow, 0.01327431072255291**loss_pow]
    criterion = CELoss(class_weights).to(gc.device)

    epoch = 0
    stop_training = False
    model.train()

    # Train Loop
    while True:
        train_data.sampler.set_epoch(epoch)

        model.train()
        
        for x, y, _ in train_data:
            opt.zero_grad()

            x, y = x.to(gc.device), y.to(gc.device)

            logits = model.forward(x)
            loss = criterion.forward(logits, y)
            loss.backward()

            opt.step()
            scheduler.step()
        
        loss_avg = loss.detach()
        if dist.is_initialized():
            dist.reduce(loss_avg, dst=0, op=dist.ReduceOp.SUM)
        loss_avg_train = loss_avg.item() / float(gc.world_size)

        # Compute score
        predictions = torch.argmax(torch.softmax(logits, 1), 1)
        iou = compute_score(predictions, y, num_classes=3)
        iou_avg = iou.detach()
        if dist.is_initialized():
            dist.reduce(iou_avg, dst=0, op=dist.ReduceOp.SUM)
        iou_avg_train = iou_avg.item() / float(gc.world_size)

        stop_training = validate(model, criterion, val_data)
        epoch += 1

        if stop_training or epoch == gc["data"]["n_epochs"]:
            break
    

    