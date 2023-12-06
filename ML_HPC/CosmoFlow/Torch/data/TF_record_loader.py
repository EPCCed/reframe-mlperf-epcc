import os

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from tfrecord.reader import tfrecord_loader

from ML_HPC.gc import GlobalContext

gc = GlobalContext()

class CosmoDataset(Dataset):
    def __init__(self, dataset_path, train=True):
        if train:
            size = gc["data"]["n_train"]
        else:
            size = gc["data"]["n_eval"]
        self.files = os.listdir(dataset_path)[:size]
        self.root = dataset_path
        self.length = len(self.files)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        # Theres only one sample per file
        data = next(tfrecord_loader(os.path.join(self.root, self.files[index]),
                                    None,
                                    description={"x": "byte", "y":"float"},
                                    compression_type=gc["data"]["compression"]
                                    ))
        x = torch.frombuffer(data["x"], dtype=torch.int16)
        x = torch.reshape(x, [128,128,128,4]).to(torch.float32)
        x = x.permute(3,0,1,2)
        y = torch.tensor(data["y"], dtype=torch.float32)

        if gc["data"]["apply_log"]:
            x = torch.log(x+1)
        else:
            x /= (torch.sum(x)/torch.prod(torch.tensor(x.shape)))
        
        return x, y
    
def get_train_dataloader():
    path = os.path.join(gc["data"]["data_dir"], "train/")
    data = CosmoDataset(path, train=True)
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
                      num_workers=1,
                      pin_memory=True if gc.device != "cpu" else False,
                      prefetch_factor=gc["data"]["prefetch"]
                      )

def get_val_dataloader():
    path = os.path.join(gc["data"]["data_dir"], "validation/")
    data = CosmoDataset(path, train=False)
    if gc.world_size > 1:
        sampler = DistributedSampler(data, gc.world_size, gc.rank, drop_last=True)
    else:
        sampler = SequentialSampler(data)
    
    local_bs = gc["data"]["global_batch_size"] // gc.world_size
    
    return DataLoader(data, 
                      sampler=sampler, 
                      batch_size=local_bs,
                      drop_last=True,
                      num_workers=1,
                      pin_memory=True if gc.device != "cpu" else False,
                      prefetch_factor=gc["data"]["prefetch"]
                      )


if __name__ == "__main__":
    ...
