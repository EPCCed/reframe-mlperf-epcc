from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import torchvision  
import torch
import poptorch
import os
import random
from ML.gc import GlobalContext

gc = GlobalContext()

def get_dummy_dataloader(options, length=1024, shape=(3,224,224)):
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
    
    
    return poptorch.DataLoader(
                    options=options,
                    dataset=dataset, 
                    batch_size = local_bs,
                    shuffle = gc["data"]["shuffle"],
                    num_workers=1,
                    drop_last = gc["data"]["drop_last_batch"]
                    )


def get_train_dataloader(options):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
        torchvision.transforms.ConvertImageDtype(torch.float16)
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

    return poptorch.DataLoader(options=options,
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

    return poptorch.DataLoader(options=options,
                               dataset=dataset,
                               batch_size = local_bs,
                               shuffle = gc["data"]["shuffle"],
                               num_workers=1,
                               drop_last = gc["data"]["drop_last_batch"]
                               )
