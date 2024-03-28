import os
import torch
from torchvision.datasets import ImageNet
import torchvision
from torchvision.io import read_image
from torchvision.datapoints import Image
from torch.utils.data import DataLoader, Dataset


def get_train_dataloader(params):
    params = params["train_input"]
    dtype = torch.float16 if params["to_float16"] else torch.float32
    transform = torchvision.transforms.Compose([
        #torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        #torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])
        #                                 
        ])
    dataset = ImageNet(root=os.path.join(os.getcwd(), params["data_dir"]),
                        split="train",
                        transform=transform,
                        )  # target_transform=target_transform)
    return DataLoader(dataset, 
                      batch_size=params["batch_size"], 
                      drop_last=params["drop_last_batch"],
                      #shuffle=params["shuffle"],
                      num_workers=params.get("num_workers", 0)
                      )

def get_val_dataloader(params):
    params = params["eval_input"]
    transform = torchvision.transforms.Compose([
        #torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                         ])
    dataset = ImageNet(root=params["data_dir"],
                       split="val",
                       transform=transform,
                    )  # target_transform=target_transform)
    return DataLoader(dataset, 
                      batch_size=params["batch_size"], 
                      drop_last=params["drop_last_batch"],
                      num_workers=params.get("num_workers", 0)
                      )

