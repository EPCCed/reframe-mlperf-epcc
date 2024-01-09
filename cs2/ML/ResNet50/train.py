import os

import torch
import yaml
import cerebras_pytorch as cstorch

from model import ResNet50
from data import get_train_dataloader, get_val_dataloader

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

        return loss

    executor = cstorch.utils.data.DataExecutor(
        cerebras_loader, num_steps=params["runconfig"]["max_steps"], checkpoint_steps=params["runconfig"]["checkpoint_steps"]
    )

    for x, y in executor:
        loss = train_step(x, y)


if __name__ == "__main__":

    cs_config = cstorch.utils.CSConfig(
        mount_dirs=[os.getcwd(), "/mnt/e1000/home/z043/z043/crae-cs1/data"],
        python_paths=[os.getcwd()],
        max_wgt_servers=1,
        num_workers_per_csx=1,
        max_act_per_csx=1,
    )

    main(cs_config)