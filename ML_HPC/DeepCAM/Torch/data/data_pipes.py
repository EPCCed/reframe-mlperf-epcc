import os

import h5py as h5
import numpy as np
import torchdata
import torch
import torchdata.dataloader2
import torchdata.dataloader2.adapter

from ML_HPC.DeepCAM.Torch import data
from ML_HPC.gc import GlobalContext

gc = GlobalContext()

DATA_SHIFT = np.array([0.02614459209144115447998046875,-88.33798980712890625,-84.620941162109375,-78.56366729736328125,-77.72217559814453125,0.00000000000733015557974336928737,48330.79296875,87595.4296875,183.576385498046875,208.382659912109375,-0.00000000000000000071859578636258,109.64270782470703125,94.19403076171875,-0.3758443892002105712890625,9952.041015625,20.362579345703125])

DATA_SCALE = np.array([0.00917674601078033447265625,0.0057406970299780368804931640625,0.00574738346040248870849609375,0.006438176147639751434326171875,0.006318948231637477874755859375,6.866295337677001953125,0.0000169723316503223031759262085,0.00004090996662853285670280456543,0.01545356027781963348388671875,0.012881465256214141845703125,26774.626953125,0.0041156332008540630340576171875,0.0042087095789611339569091796875,0.0001746262423694133758544921875,0.00033861628617160022258758544922,0.01948749832808971405029296875])

def preprocess_fn(filename):
    with h5.File(filename, "r") as f:
        data = f["climate"]["data"][...]
        label = f["climate"]["labels_0"][...].astype(np.int64)
    
    data = DATA_SCALE * (data - DATA_SHIFT)
    data = np.transpose(data, (2,0,1)).astype(np.float32)
    return torch.from_numpy(data), torch.from_numpy(label)


def get_dataloaders():
    local_bs = gc["data"]["global_batch_size"] // gc.world_size
    if gc["data"]["gradient_accumulation_freq"] == -1:
        if local_bs > 64:
            gc["data"]["gradient_accumulation_freq"] = local_bs // 64

            local_bs = local_bs // gc["data"]["gradient_accumulation_freq"]
        else:
            gc["data"]["gradient_accumulation_freq"] = 1
    else:
        local_bs = local_bs // gc["data"]["gradient_accumulation_freq"]
    
    if gc["data"]["local_batch_size"]:
        gc["data"]["gradient_accumulation_freq"] = 1
        local_bs = gc["data"]["local_batch_size"]
        gc["data"]["global_batch_size"] = gc.world_size * local_bs

    print(local_bs)
    source = os.path.join(gc["data"]["data_dir"], "train")
    data_pipe = torchdata.datapipes.iter.FileLister(source)
    data_pipe = data_pipe.sharding_filter().shuffle()
    data_pipe = data_pipe.map(preprocess_fn).batch(local_bs)
    rs = torchdata.dataloader2.DistributedReadingService()
    train_dl = torchdata.dataloader2.DataLoader2(data_pipe, reading_service=rs)
    train_size = len(os.listdir(source))

    source = os.path.join(gc["data"]["data_dir"], "validation")
    data_pipe = torchdata.datapipes.iter.FileLister(source)
    data_pipe = data_pipe.sharding_filter()
    data_pipe = data_pipe.map(preprocess_fn).batch(1)
    rs = torchdata.dataloader2.DistributedReadingService()
    val_dl = torchdata.dataloader2.DataLoader2(data_pipe, reading_service=rs)
    val_size = len(os.listdir(source))

    return train_dl, train_size, val_dl, val_size


