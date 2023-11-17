import os

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler
from tfrecord.reader import tfrecord_loader

from ML.gc import GlobalContext

gc = GlobalContext()

class CosmoDataset(Dataset):
    def __init__(self, dataset_path, train=True):
        if train:
            size = gc["data"]["n_train"]
        else:
            size = gc["data"]["n_eval"]
        self.files = os.listdir(dataset_path)[:size]
        self.root - dataset_path
        self.length = len(self.files)
    
    def len(self):
        return self.length
    
    def __getitem__(self, index):
        # Theres only one sample per file
        data = next(tfrecord_loader(os.path.join(self.root, self.files[index]),
                                    description={"x": "byte", "y":"float"},
                                    compression_type=gc["data"]["compression"]
                                    ))
        x = torch.frombuffer(data["x"], dtype=torch.int16)
        x = torch.reshape(x, [128,128,128,4]).to(torch.float32)
        y = torch.tensor(data["y"], dtype="float32")

        if gc["data"]["apply_log"]:
            x = torch.log(x+1)
        else:
            x /= (torch.sum(x)/torch.prod(torch.tensor(x.shape)))
        
        return x, y
    
def get_train_dataloader():
    path = os.pat.join(gc["data"]["data_dir"], "train/")
    data = CosmoDataset(path, train=True)
    if gc.world_size > 1:
        sampler = DistributedSampler(data, gc.world_size, gc.rank, shuffle=gc["data"]["shuffle"], drop_last=True)
    else:
        sampler = RandomSampler(data)
    local_bs = gc["data"]["global_batch_size"] // gc.world_size
    return DataLoader(data, 
                      sampler=sampler, 
                      batch_size=local_bs, 
                      shuffle=gc["data"]["shuffle"],
                      drop_last=True,
                      pin_memory=True if gc.device != "cpu" else False
                      )

def get_val_dataloader():
    path = os.pat.join(gc["data"]["data_dir"], "validation/")
    data = CosmoDataset(path, train=False)
    if gc.world_size > 1:
        sampler = DistributedSampler(data, gc.world_size, gc.rank, shuffle=gc["data"]["shuffle"], drop_last=True)
    else:
        sampler = RandomSampler(data)
    local_bs = gc["data"]["global_batch_size"] // gc.world_size
    return DataLoader(data, 
                      sampler=sampler, 
                      batch_size=local_bs, 
                      shuffle=gc["data"]["shuffle"],
                      drop_last=True,
                      pin_memory=True if gc.device != "cpu" else False)


if __name__ == "__main__":
    ...
