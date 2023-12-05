from pathlib import Path
import sys
import os
path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))

import torch
import torch.distributed as dist
import colossalai
from colossalai.lazy import LazyInitContext
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin

from ML.gc import GlobalContext
gc = GlobalContext("/work/ta127/ta127/chrisrae/chris-ml-intern/ML/GPT3/Torch/config.yaml")
from ML.GPT3.Torch.model.gpt3 import get_gpt3_175b

def main():
    if dist.is_mpi_available():
        backend = "mpi"
    elif gc.device == "cuda":
        backend = "nccl"
    else:
        backend = "gloo"
    colossalai.launch_from_slurm(config={}, 
                                 host=...,
                                 port=29500,
                                 backend=backend,
                                 seed=0)
    
    train_dataloader, val_dataloader = ...

    with LazyInitContext(default_device=gc.device):
        model = get_gpt3_175b(vocab_size=...,pad_idx=...)
    
    if gc["opt"]["name"].upper() == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=gc["lr_schedule"]["base_lr"],
                                    betas=gc["opt"]["betas"],
                                    weight_decay=gc["opt"]["weight_decay"])
    else:
        raise NameError(f"Optimiser {gc['opt']['name']} not supported please use ADAM")

    lr_scheduler = ...

    criterion = ...
    
    plugin = HybridParallelPlugin(tp_size=gc["parallel"]["tp_size"],
                                  pp_size=gc["parallel"]["pp_size"],
                                  )

    


if __name__ == "__main__":
    main()