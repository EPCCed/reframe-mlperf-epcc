# The MIT License (MIT)
#
# Copyright (c) 2018 Pyjcsx
# Modifications Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import glob
import h5py as h5
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from ML_HPC.gc import GlobalContext

gc = GlobalContext()

def peek_shapes_hdf5(data_dir):
    files = glob.iglob(os.path.join(data_dir, "*.h5"))
    with h5.File(next(files), "r") as fin:
        data_shape = fin["climate"]["data"].shape
        label_shape = fin["climate"]["labels_0"].shape
        
    return data_shape, label_shape


#dataset class
class CamDataset(Dataset):
  
    def init_reader(self):
        #shuffle
        if self.shuffle:
            self.rng.shuffle(self.all_files)
            
        #shard the dataset
        self.global_size = len(self.all_files)
        if self.allow_uneven_distribution:
            # this setting covers the data set completely

            # deal with bulk
            num_files_local = self.global_size // self.comm_size
            start_idx = self.comm_rank * num_files_local
            end_idx = start_idx + num_files_local
            self.files = self.all_files[start_idx:end_idx]

            # deal with remainder
            for idx in range(self.comm_size * num_files_local, self.global_size):
                if idx % self.comm_size == self.comm_rank:
                    self.files.append(self.all_files[idx])
        else:
            # here, every worker gets the same number of samples, 
            # potentially under-sampling the data
            num_files_local = self.global_size // self.comm_size
            start_idx = self.comm_rank * num_files_local
            end_idx = start_idx + num_files_local
            self.files = self.all_files[start_idx:end_idx]
            self.global_size = self.comm_size * len(self.files)
            
        #my own files
        self.local_size = len(self.files)

        #print sizes
        #print("Rank {} local size {} (global {})".format(self.comm_rank, self.local_size, self.global_size))

  
    def __init__(self, source, statsfile, channels,
                 allow_uneven_distribution = False,
                 shuffle = False,
                 preprocess = True,
                 transpose = True,
                 comm_size = 1, comm_rank = 0, seed = 12345):
        
        self.source = source
        self.statsfile = statsfile
        self.channels = channels
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.transpose = transpose
        self.all_files = sorted( [ os.path.join(self.source,x) for x in os.listdir(self.source) if x.endswith('.h5') and x != "stats.h5"] )
        self.comm_size = comm_size
        self.comm_rank = comm_rank
        self.allow_uneven_distribution = allow_uneven_distribution
        
        #split list of files
        self.rng = np.random.RandomState(seed)
        
        #init reader
        self.init_reader()

        #get shapes
        filename = os.path.join(self.source, self.files[0])
        with h5.File(filename, "r") as fin:
                self.data_shape = fin['climate']['data'].shape
                self.label_shape = fin['climate']['labels_0'].shape
        
        #get statsfile for normalization
        #open statsfile
        with h5.File(self.statsfile, "r") as f:
            data_shift = f["climate"]["minval"][self.channels]
            data_scale = 1. / ( f["climate"]["maxval"][self.channels] - data_shift )

        #reshape into broadcastable shape
        self.data_shift = np.reshape( data_shift, (1, 1, data_shift.shape[0]) ).astype(np.float32)
        self.data_scale = np.reshape( data_scale, (1, 1, data_scale.shape[0]) ).astype(np.float32)

    def __len__(self):
        return self.local_size


    @property
    def shapes(self):
        return self.data_shape, self.label_shape


    def __getitem__(self, idx):
        filename = os.path.join(self.source, self.files[idx])

        #load data and project
        torch.cuda.nvtx.range_push("reading file")
        with h5.File(filename, "r", rdcc_nbytes=1048576*50, rdcc_nslots=16) as f:
            data = f["climate"]["data"][..., self.channels]
            label = f["climate"]["labels_0"][...].astype(np.int64)
        torch.cuda.nvtx.range_pop()
        #preprocess
        torch.cuda.nvtx.range_push("preprocessing")
        data = self.data_scale * (data - self.data_shift)

        if self.transpose:
            #transpose to NCHW
            data = np.transpose(data, (2,0,1))
        torch.cuda.nvtx.range_pop()

        return torch.from_numpy(data), torch.from_numpy(label)

class DummyDataset(Dataset):
    def __init__(self, n_samples):
        self.n_samples = n_samples
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.ones(16, 768, 1152).to(gc.device),  torch.ones(3, 768, 1152).to(gc.device), "file.txt"


def get_datashapes():
    return peek_shapes_hdf5(os.path.join(gc["data"]["data_dir"], "train"))


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
    
    # import only what we need
    train_dir = os.path.join(gc["data"]["data_dir"], "train")
    train_set = CamDataset(train_dir, 
                           statsfile = os.path.join(gc["data"]["data_dir"], 'stats.h5'),
                           channels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                           allow_uneven_distribution = False,
                           shuffle = True, 
                           preprocess = True,
                           comm_size = 1,
                           comm_rank = 0)
    
    distributed_train_sampler = DistributedSampler(train_set,
                                                   num_replicas = gc.world_size,
                                                   rank = gc.rank,
                                                   shuffle = True,
                                                   drop_last = True)
    
    train_loader = DataLoader(train_set,
                              batch_size=local_bs,
                              num_workers =4,
                              sampler = distributed_train_sampler,
                              pin_memory = True if gc.device != "cpu" else False,
                              drop_last = True,
    prefetch_factor=gc["data"]["prefetch"])

    train_size = train_set.global_size

    validation_dir = os.path.join(gc["data"]["data_dir"], "validation")
    validation_set = CamDataset(validation_dir, 
                                statsfile = os.path.join(gc["data"]["data_dir"], 'stats.h5'),
                                channels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                                allow_uneven_distribution = True,
                                shuffle = False,
                                preprocess = True,
                                comm_size = gc.world_size,
                                comm_rank = gc.rank)
    
    # use batch size = 1 here to make sure that we do not drop a sample
    validation_loader = DataLoader(validation_set,
                                   batch_size=1,
                                   num_workers = 8,
                                   pin_memory = True if gc.device != "cpu" else False,
                                   drop_last = False,
    prefetch_factor=gc["data"]["prefetch"])
    
    validation_size = validation_set.global_size    
        
    return train_loader, train_size, validation_loader, validation_size

if __name__ == "__main__":
    import time
    from tqdm import tqdm
    gc.update_config("/workspace/ML_HPC/DeepCAM/Torch/config.yaml")
    gc.init_dist()
    train_loader, train_size, validation_loader, validation_size = get_dataloaders()
    
    train_loader = tqdm(train_loader)
 
    
    for E in range(gc["data"]["n_epochs"]):
        start = time.time()
        iter_time = 0
        host_to_dev_time = 0
    
        i = 0
    
    
        t0 = time.time_ns()
        
        for x, y in train_loader:
            t1 = time.time_ns()
            x, y = x.cuda(), y.cuda()
            torch.cuda.synchronize()
            iter_time += t1 - t0
            host_to_dev_time += time.time_ns() - t1
            t0 = time.time_ns()

            i+=1
        print(f"Total Time: {time.time()- start}")
        print(f"Iterable Time: {iter_time*1e-9}")
        print(f"Host To Device Time: {host_to_dev_time*1e-9}")