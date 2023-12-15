from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import torchvision  
import poptorch
import os
import random
from ML.gc import GlobalContext
from ML_HPC.DeepCAM.Torch import data

gc = GlobalContext()


def get_train_dataloader(options):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                         ])
    dataset = ImageFolder(root=os.path.join(gc["data"]["data_dir"], "train"),
                          transform=transform)
    
    if gc["data"]["train_subset"]:
        indices = random.sample(range(len(dataset)), gc["data"]["train_subset"])
        dataset = Subset(dataset, indices)

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

    return poptorch.Dataloader(options=options,
                               dataset=dataset,
                               batch_size = local_bs,
                               shuffle = gc["data"]["shuffle"],
                               num_workers=1,
                               drop_last = gc["data"]["drop_last_batch"]
                               )


def get_val_dataloader(options):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                         ])
    dataset = ImageFolder(root=os.path.join(gc["data"]["data_dir"], "val"),
                          transform=transform)
    if gc["data"]["val_subset"]:
        indices = random.sample(range(len(dataset)), gc["data"]["val_subset"])
        dataset = Subset(dataset, indices)
    
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

    return poptorch.Dataloader(options=options,
                               dataset=dataset,
                               batch_size = local_bs,
                               shuffle = gc["data"]["shuffle"],
                               num_workers=1,
                               drop_last = gc["data"]["drop_last_batch"]
                               )
