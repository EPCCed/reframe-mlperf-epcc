import os
import random

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import torchvision

from ML.gc import GlobalContext

gc = GlobalContext()



def get_train_dataloader():

    path = os.path.join(gc["data"]["data_dir"], "train")

    local_bs = gc["data"]["global_batch_size"] // gc.world_size
    if gc["data"]["gradient_accumulation_freq"] == -1:
        if local_bs > 64:
            gc["data"]["gradient_accumulation_freq"] = local_bs // 64

            local_bs = local_bs // gc["data"]["gradient_accumulation_freq"]
        else:
            gc["data"]["gradient_accumulation_freq"] = 1
    else:
        local_bs = local_bs // gc["data"]["gradient_accumulation_freq"]

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                         ])
    dataset = ImageFolder(root=path,
                          transform=transform)

    if gc["data"]["train_subset"]:
        indices = random.sample(range(len(dataset)), gc["data"]["train_subset"])
        dataset = Subset(dataset, indices)

    if gc.world_size > 1:
        sampler = DistributedSampler(dataset, gc.world_size, gc.rank, shuffle=gc["data"]["shuffle"])
    else:
        if gc["data"]["shuffle"]:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

    
    if gc["data"]["local_batch_size"]:
        gc["data"]["gradient_accumulation_freq"] = 1
        local_bs = gc["data"]["local_batch_size"]
        gc["data"]["global_batch_size"] = gc.world_size * local_bs

    return DataLoader(dataset, 
                      sampler=sampler,
                      batch_size=local_bs, 
                      drop_last=gc["data"]["drop_last_batch"],
                      num_workers=4,
                      prefetch_factor=gc["data"]["prefetch"],
                      pin_memory = True if gc.device == "cuda" else False 
                      )

def get_val_dataloader():
    
    path = os.path.join(gc["data"]["data_dir"], "val")

    local_bs = gc["data"]["global_batch_size"] // gc.world_size

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                         ])
    dataset = ImageFolder(root=path,
                          transform=transform)
    if gc["data"]["val_subset"]:
        indices = random.sample(range(len(dataset)), gc["data"]["val_subset"])
        dataset = Subset(dataset, indices)
    if gc.world_size > 1:
        sampler = DistributedSampler(dataset, gc.world_size, gc.rank)
    else:
        if gc["data"]["shuffle"]:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
    local_bs = gc["data"]["global_batch_size"] // gc.world_size
    return DataLoader(dataset, 
                      sampler=sampler,
                      batch_size=local_bs, 
                      drop_last=gc["data"]["drop_last_batch"],
                      num_workers=1,
                      prefetch_factor=gc["data"]["prefetch"],
                      pin_memory = True if gc.device == "cuda" else False 
                      )

