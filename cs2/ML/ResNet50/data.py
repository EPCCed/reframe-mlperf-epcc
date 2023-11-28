import os
import torch
from torchvision.datasets import ImageFolder
import torchvision
from torchvision.io import read_image
from torchvision.datapoints import Image
from torch.utils.data import DataLoader, Dataset

class ImageNet(Dataset):
    def __init__(self, root):
        self.root = root
        self.dirs = os.listdir(self.root)
        self.dirs.sort()
        self.transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                         ])
        self.label_table = {n: i for i,n in enumerate(self.dirs)}
    
    def __len__(self):
        return 1000000
    
    def __getitem__(self, idx):
        dir_idx = 1000000//idx
        img_idx = idx%1000
        _dir = self.dirs[dir_idx]
        img_name = os.listdir(os.path.join(self.root, _dir))[img_idx]
        path = os.path.join(self.root, _dir, img_name)
        x = Image(read_image(path))
        y = self.label_table[_dir]
        return x, y



def get_train_dataloader(params):
    params = params["train_input"]
    dtype = torch.float16 if params["to_float16"] else torch.float32
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                         ])
    target_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = ImageFolder(root=params["data_dir"],
                          transform=transform,
                          target_transform=target_transform)
    return DataLoader(dataset, 
                      batch_size=params["batch_size"], 
                      drop_last=params["drop_last_batch"],
                      shuffle=params["shuffle"],
                      num_workers=params.get("num_workers", 0)
                      )

def get_val_dataloader(params):
    params = params["eval_input"]
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                         ])
    target_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = ImageFolder(root=params["data_dir"],
                          transform=transform,
                          target_transform=target_transform)
    return DataLoader(dataset, 
                      batch_size=params["batch_size"], 
                      drop_last=params["drop_last_batch"],
                      num_workers=params.get("num_workers", 0)
                      )

