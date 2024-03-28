import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler
import h5py

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
    
def get_train_dataloader(params):
    params = params["train_input"]
    data = CosmoDataset(params["data_dir"], train=True)
    return DataLoader(data, 
                      batch_size=params["batch_size"], 
                      drop_last=params["drop_last_batch"],
                      shuffle=params["shuffle"],
                      num_workers=params.get("num_workers", 0)
                      )

def get_val_dataloader(params):
    params = params["eval_input"]
    data = CosmoDataset(params["data_dir"], train=True)
    return DataLoader(data, 
                      batch_size=params["batch_size"], 
                      drop_last=params["drop_last_batch"],
                      num_workers=params.get("num_workers", 0)
                      )

if __name__ == "__main__":
    import yaml
    with open("./params.yaml", "r") as stream:
        params = yaml.safe_load(stream)
    loader = get_train_dataloader(params)
    for x,y in loader:
        print(x.shape, y.shape)
        break
    