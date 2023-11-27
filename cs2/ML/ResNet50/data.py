from torchvision.datasets import ImageFolder
import torchvision
from torch.utils.data import DataLoader

def get_train_dataloader(params):
    params = params["train_input"]
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                         ])
    dataset = ImageFolder(root=params["data_dir"],
                          transform=transform)
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
    dataset = ImageFolder(root=params["data_dir"],
                          transform=transform)
    return DataLoader(dataset, 
                      batch_size=params["batch_size"], 
                      drop_last=params["drop_last_batch"],
                      num_workers=params.get("num_workers", 0)
                      )

