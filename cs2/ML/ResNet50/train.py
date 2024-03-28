import os
import time

import cerebras_pytorch as cstorch
import torch
import yaml

from model import ResNet50
#from data import get_train_dataloader, get_val_dataloader

def get_train_dataloader(params):
    import torch
    from torchvision.datasets import ImageFolder
    import torchvision
    from torch.utils.data import DataLoader, Dataset
    import os

    params = params["train_input"]
    dtype = torch.float16 if params["to_float16"] else torch.float32
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                         ])
    #target_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = ImageFolder(root=os.path.join(os.getcwd(), params["data_dir"]),
                          transform=transform,
                          )  # target_transform=target_transform
    return DataLoader(dataset, 
                      batch_size=params["batch_size"], 
                      drop_last=params["drop_last_batch"],
                      shuffle=params["shuffle"],
                      num_workers=8
                      )

def get_dummy_dataloader(params, length=65536, shape=(3,224,224)):
    import torch
    from torch.utils.data import DataLoader, Dataset
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
            label = torch.zeros(1000)
            label[torch.randint(0, 999, (1,))] = 1
            return img, label
    
    params: dict = params["train_input"]
    dataset = DummyData(length, shape)
    return DataLoader(dataset, 
                      batch_size=params["batch_size"], 
                      drop_last=params["drop_last_batch"],
                      shuffle=params["shuffle"],
                      prefetch_factor=4,
                      num_workers=params.get("num_workers", 0)
                      )

def main(config):
    with open("./params.yaml", "r") as stream:
        params = yaml.safe_load(stream)
    
    model = ResNet50()
    compiled_model = cstorch.compile(model, backend="CSX")

    optimizer = cstorch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    cerebras_loader = cstorch.utils.data.DataLoader(get_train_dataloader, params)

    criterion = torch.nn.CrossEntropyLoss()
    
    @cstorch.trace
    def train_step(x, y):
        logits = compiled_model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        cstorch.summarize_scalar("loss", loss)
        return loss

    writer = cstorch.utils.tensorboard.SummaryWriter()
    executor = cstorch.utils.data.DataExecutor(
        cerebras_loader, num_steps=params["runconfig"]["max_steps"], checkpoint_steps=params["runconfig"]["checkpoint_steps"], cs_config=cs_config, writer=writer
    )
    start = time.time()
    for x, y in executor:
        loss = train_step(x, y)
    total_time = time.time()-start
    dataset_size = len(cerebras_loader)*params["train_input"]["batch_size"]
    
    print(f"Processing Speed: {dataset_size/total_time}")
    print(f"Time For Epoch: {total_time}")

def get_default_inis():
    return {
        "ws_cv_auto_inis": True,
        "ws_opt_target_max_actv_wios": 16,
        "ws_opt_max_wgt_port_group_size": 8,
        "ws_run_memoize_actv_mem_mapping": True,
        "ws_opt_bidir_act": False,
        "ws_variable_lanes": True,
    }

if __name__ == "__main__":
    #debug_args = DebugArgs()
    #set_default_ini(debug_args, **get_default_inis())
    #write_debug_args(debug_args, os.path.join(os.getcwd(), ".debug_args.proto"))

    cs_config = cstorch.utils.CSConfig(
        mount_dirs=["/mnt/e1000/home/z043/z043/crae-cs1/chris-ml-intern/cs2/modelzoo", os.getcwd()],
        python_paths=[os.getcwd()],
        #max_wgt_servers=1,
        num_workers_per_csx=8,
        #max_act_per_csx=1,
        num_csx=1
        #debug_args=debug_args
    )

    main(cs_config)
    """
    from functools import partial
    with open("./params.yaml", "r") as stream:
        params = yaml.safe_load(stream)
    cstorch.utils.benchmark.benchmark_dataloader(
    input_fn=partial(get_train_dataloader, params),
    num_epochs=2,
    steps_per_epoch=1000,
    )
    """
