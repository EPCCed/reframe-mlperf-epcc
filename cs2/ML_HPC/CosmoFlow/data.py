import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler
from tfrecord.reader import tfrecord_loader

class CosmoDataset(Dataset):
    def __init__(self, dataset_path, train=True):
        self.files = os.listdir(dataset_path)
        self.root = dataset_path
        self.length = len(self.files)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        # Theres only one sample per file
        data = next(tfrecord_loader(os.path.join(self.root, self.files[index]),
                                    None,
                                    description={"x": "byte", "y":"float"},
                                    compression_type="gzip"
                                    ))
        x = torch.frombuffer(data["x"], dtype=torch.int16)
        x = torch.reshape(x, [128,128,128,4]).to(torch.float32)
        x = x.permute(3,0,1,2)
        y = torch.tensor(data["y"], dtype=torch.float32)

        if False:
            x = torch.log(x+1)
        else:
            x /= (torch.sum(x)/torch.prod(torch.tensor(x.shape)))
        
        return x, y
    
def get_train_dataloader(params):
    params = params["train_input"]
    data = CosmoDataset(params["data_dir"], train=True)
    return DataLoader(data, 
                      batch_size=params["batch_size"], 
                      drop_last=params["drop_last_batch"],
                      shuffle=params["shuffle"],
                      num_workers=params.get("num_workers", 0)
                      )

def get_val_dataloader():
    params = params["eval_input"]
    data = CosmoDataset(params["data_dir"], train=True)
    return DataLoader(data, 
                      batch_size=params["batch_size"], 
                      drop_last=params["drop_last_batch"],
                      num_workers=params.get("num_workers", 0)
                      )
