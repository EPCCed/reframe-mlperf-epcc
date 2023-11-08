from itertools import cycle, chain  # correct batch sizing
import os
from typing import Any
from threading import Thread

from mpi4py import MPI
import torch

def chunk_files_per_worker(files: list, n_workers: int):
    n = len(files) // n_workers 
    int_chunk = [files[x:x+n] for x in range(0, len(files), n)]
    if len(int_chunk) == n_workers + 1:
        remainder = int_chunk.pop()
        for i,r in enumerate(remainder):
            int_chunk[i].append(r)
    return int_chunk

class CosmoData():
    def __init__(self, file_path, train=True, batch_size=64, shuffle=False):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()
        self.samples_per_file = 64
        self.w_p = 0  # sliding window postion used for buffer
    
        self.file_path = os.path.join(file_path, "train/" if train else "test/")
        self.files = cycle(chunk_files_per_worker(os.listdir(self.file_path), world_size)[rank])
        self.pwbs = batch_size // world_size  # per worker batch size
        self.buffersize = 64 if self.pwbs <= 64 else self.pwbs
        self.data = torch.empty(self.buffersize*2, 4, 128, 128,128)
        self.labels = torch.empty(self.buffersize*2, 4)
        self.data[:self.buffersize], self.labels[:self.buffersize]  = self._process_file()
        self.data[self.buffersize:], self.labels[self.buffersize:]  = self._process_file()  # buffer

    def _process_file(self):
        bd = []
        bl = []
        for _ in range(self.buffersize//self.samples_per_file):
            p = os.path.join(self.file_path, next(self.files))
            d, l = self._read_file(p)
            bd.append(d)
            bl.append(l)
        return torch.concatenate(bd, 0), torch.concatenate(bl,0)

    def _read_file(self, file):
        file_out = torch.load(file)
        return file_out["data"], file_out["label"].repeat(64,1)  # not memory effecient
    
    def __next__(self):
        
        if hasattr(self, 'thread'):
            self.thread.join()

        next_data = self.data[self.w_p*self.pwbs:(self.w_p+1)*self.pwbs]
        next_label = self.labels[self.w_p*self.pwbs:(self.w_p+1)*self.pwbs]

        if self.buffersize == self.pwbs*(self.w_p+1):
            def _set_buffer_0():
                self.data[:self.buffersize], self.labels[:self.buffersize]  = self._process_file()
            Thread(target=_set_buffer_0).start()
            self.w_p += 1
        if self.buffersize*2 == self.pwbs*(self.w_p+1):
            def _set_buffer_1():
                self.data[self.buffersize:], self.labels[self.buffersize:]   = self._process_file()
            Thread(target=_set_buffer_1).start()
            self.w_p =0
        else:
            self.w_p += 1
    
        """
        next_value = self.data

        def _load_next_buffer():
            self.data = self.buffer
            self.buffer = self._load_batch()

        self.thread = Thread(target=_load_next_buffer)
        self.thread.start()
        """
        return next_data, next_label
    
    def __iter__(self):
        return self



if __name__ == '__main__':
    for i,j in CosmoData("/Users/chrisrae/Projects/chris-ml-intern/", batch_size=128):
        print(i.shape, j.shape)
        exit()
    ...