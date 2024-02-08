import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
import h5py

from ML_HPC.gc import GlobalContext
gc = GlobalContext()

class CosmoDataset(Dataset):
    def __init__(self, dataset_path, train=True):
        self.files = os.listdir(dataset_path)
        self.root = dataset_path
        self.length = len(self.files)

    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        with h5py.File(os.path.join(self.root, self.files[index]), "r") as hf:
            x = torch.from_numpy(hf["x"][:])
            y = torch.from_numpy(hf["y"][:])
        if False:
            x = torch.log(x+1)
        else:
            x /= (torch.sum(x)/torch.prod(torch.tensor(x.shape)))
        
        return x.to(torch.float16), y.to(torch.float16)
    
def get_train_dataloader():
    data = CosmoDataset(os.path.join(gc["data"]["data_dir"], "hdf5_train"), train=True)
    if gc.world_size > 1:
        sampler = DistributedSampler(data, gc.world_size, gc.rank, shuffle=gc["data"]["shuffle"], drop_last=True)
    else:
        if gc["data"]["shuffle"]:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
    local_bs = gc["data"]["global_batch_size"] // gc.world_size
    if gc["data"]["gradient_accumulation_freq"] == -1:
        if local_bs > 16:
            gc["data"]["gradient_accumulation_freq"] = local_bs // 16
            local_bs = local_bs // gc["data"]["gradient_accumulation_freq"]
        else:
            gc["data"]["gradient_accumulation_freq"] = 1
    
    if gc["data"]["local_batch_size"]:
        gc["data"]["gradient_accumulation_freq"] = 1
        local_bs = gc["data"]["local_batch_size"]
        gc["data"]["global_batch_size"] = gc.world_size * local_bs
    return DataLoader(data, 
                      sampler=sampler, 
                      batch_size=local_bs, 
                      drop_last=True,
                      num_workers=4,
                      pin_memory=True if gc.device != "cpu" else False,
                      prefetch_factor=gc["data"]["prefetch"]
                      )

def get_val_dataloader():
    data = CosmoDataset(os.path.join(gc["data"]["data_dir"], "hdf5_val"), train=True)
    if gc.world_size > 1:
        sampler = DistributedSampler(data, gc.world_size, gc.rank, drop_last=True)
    else:
        sampler = SequentialSampler(data)
    
    local_bs = gc["data"]["global_batch_size"] // gc.world_size
    
    return DataLoader(data, 
                      sampler=sampler, 
                      batch_size=local_bs,
                      drop_last=True,
                      num_workers=4,
                      pin_memory=True if gc.device != "cpu" else False,
                      prefetch_factor=gc["data"]["prefetch"]
                      )