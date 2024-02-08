import os
from random import shuffle
from scipy import datasets

import torch
import poptorch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler
#from tfrecord.reader import tfrecord_loader

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


def get_train_dataloader(options):
    path = os.path.join(gc["data"]["data_dir"], "train/")
    data = CosmoDataset(path, train=True)
    
    local_bs = gc["data"]["global_batch_size"] //gc["training"]["num_ipus"]
    iters_per_epoch = len(os.listdir(path)) // local_bs
    options.deviceIterations(iters_per_epoch)

    return poptorch.DataLoader(options=options,
                               dataset=data,
                               batch_size = local_bs,
                               shuffle=gc["data"]["shuffle"],
                               num_workers=1,
                               drop_last=True
                               )


def get_val_dataloader(options):
    path = os.path.join(gc["data"]["data_dir"], "validation/")
    data = CosmoDataset(path, train=False)
    local_bs = gc["data"]["global_batch_size"] //gc["training"]["num_ipus"]
    iters_per_epoch = len(os.listdir(path)) // local_bs
    options.deviceIterations(iters_per_epoch)

    return poptorch.DataLoader(options=options,
                               dataset=data,
                               batch_size = local_bs,
                               shuffle=gc["data"]["shuffle"],
                               num_workers=1,
                               drop_last=True)

def get_dummy_dataloader(options, length=1024, shape=(4,128,128,128)):
    class DummyData(Dataset):
        def __init__(self, length, shape):
            self.length = length
            self.shape = shape
            self.img = torch.randint(0, 255, self.shape, dtype=torch.float32) / 255.0
            self.label = torch.zeros(1000)
            self.label[torch.randint(0, 999, (1,))] = 1
    
        def __len__(self):
            return self.length

        def __getitem__(self, index):
            img = torch.randint(0, 255, self.shape, dtype=torch.float32) / 255.0
            label = torch.randint(0, 999, (1,), dtype=torch.int32).squeeze()
            return img, label
    
    dataset = DummyData(length, shape)

    local_bs = gc["data"]["global_batch_size"] // gc.world_size
    if gc["data"]["gradient_accumulation_freq"] == -1:
        if local_bs > 128:
            gc["data"]["gradient_accumulation_freq"] = local_bs // 128
            local_bs = local_bs // gc["data"]["gradient_accumulation_freq"]
        else:
            gc["data"]["gradient_accumulation_freq"] = 1
    
    if gc["data"]["local_batch_size"]:
        gc["data"]["gradient_accumulation_freq"] = 1
        local_bs = gc["data"]["local_batch_size"]
        gc["data"]["global_batch_size"] = gc.world_size * local_bs
    
    return poptorch.DataLoader(
                    options=options,
                    dataset=dataset, 
                    batch_size = local_bs,
                    shuffle = gc["data"]["shuffle"],
                    num_workers=1,
                    drop_last = gc["data"]["drop_last_batch"]
                    )

if __name__ == "__main__":
    ...
