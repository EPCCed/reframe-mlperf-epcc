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
        self.all_files = sorted( [ os.path.join(self.source,x) for x in os.listdir(self.source) if x.endswith('.h5') and "stats" not in x] )
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
            data_shift = torch.from_numpy(f["climate"]["minval"][self.channels])
            data_scale = 1. / ( torch.from_numpy(f["climate"]["maxval"][self.channels]) - data_shift )

        #reshape into broadcastable shape
        self.data_shift = torch.reshape( data_shift, (1, 1, data_shift.shape[0]) ).to(torch.float32)
        self.data_scale = torch.reshape( data_scale, (1, 1, data_scale.shape[0]) ).to(torch.float32)

        if comm_rank == 0:
            print("Initialized dataset with ", self.global_size, " samples.")
        
        loss_pow = -0.125
        self.weights = torch.tensor([0.986267818390377**loss_pow, 0.0004578708870701058**loss_pow, 0.01327431072255291**loss_pow], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)


    def __len__(self):
        return self.local_size


    @property
    def shapes(self):
        return self.data_shape, self.label_shape


    def __getitem__(self, idx):
        filename = os.path.join(self.source, self.files[idx])

        #load data and project
        with h5.File(filename, "r") as f:
            data = torch.from_numpy(f["climate/data"][..., self.channels])
            label = torch.from_numpy(f["climate/labels_0"][...]).to(torch.int64)
        
        #preprocess
        data = self.data_scale * (data - self.data_shift)

        if self.transpose:
            #transpose to NCHW
            data = torch.permute(data, (2,0,1))
        
        return data, label, self.weights



def get_train_dataloader(params):
    params = params["train_input"]
    train_set = CamDataset(params["data_dir"], 
                           statsfile = os.path.join(params["data_dir"], 'stats.h5'),
                           channels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                           allow_uneven_distribution = False,
                           shuffle = True, 
                           preprocess = True,
                           comm_size = 1,
                           comm_rank = 0)
    
    
    train_loader = DataLoader(train_set,
                              batch_size=params["batch_size"],
                              pin_memory = False,
                              shuffle = params["shuffle"],
                              drop_last = params["drop_last_batch"],
                              num_workers=params.get("num_workers", 0))
    
    return train_loader


def get_eval_dataloader(params):
    params = params["train_input"]
    validation_set = CamDataset(params["data_dir"], 
                                statsfile = os.path.join(params["data_dir"], 'stats.h5'),
                                channels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                                allow_uneven_distribution = True,
                                shuffle = False,
                                preprocess = True,
                                comm_size = 1,
                                comm_rank = 0)
    
    validation_loader = DataLoader(validation_set,
                                   batch_size=params["batch_size"],
                                   pin_memory = False,
                                   drop_last = params["drop_last_batch"],
                                   num_workers=params.get("num_workers", 0))
        
    return validation_loader

if __name__ == "__main__":
    import yaml
    with open("/home/z043/z043/crae-cs1/chris-ml-intern/cs2/ML_HPC/DeepCAM/params.yaml", "r") as stream:
        params = yaml.safe_load(stream)
    loader = get_train_dataloader(params)
    for x,y,w in loader:
        #torch.Size([1, 16, 768, 1152]) torch.Size([1, 768, 1152])
        print(x.shape, y.shape)
        break
