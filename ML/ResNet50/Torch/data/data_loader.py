from torch.utils.data import DataLoader
import torchvision
import os
from ML.gc import GlobalContext

gc = GlobalContext()

def get_imagenet_dataloader(data_partition="train/", **kwargs):
    data_path = os.path.join(gc["data"]["data_dir"], data_partition)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    ])


    dataset = torchvision.datasets.ImageFolder(data_path, transform=transform)
    return DataLoader(dataset=dataset, 
                      shuffle=gc["data"]["shuffle"], 
                      batch_size=gc["data"]["global_batch_size"] // gc.world_size,
                      num_workers = gc.world_size if gc.world_size > 1 else 0,
                      drop_last=True,
                      prefetch_factor=gc["data"]["prefetch"]
                      )