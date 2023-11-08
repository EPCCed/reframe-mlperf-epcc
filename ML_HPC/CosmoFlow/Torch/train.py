
import torch 
import torch.nn as nn
import torch.distributed as dist
from mpi4py import MPI

from model.cosmoflow import StandardCosmoFlow
from ML_HPC.CosmoFlow.Torch.data.CPU_data_loader import CosmoData

LR = 0.001

def main():
    dist.init_process_group("mpi", )
    

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    model = nn.parallel.DistributedDataParallel(StandardCosmoFlow())
    opt = torch.optim.SGD(model.parameters, LR, momentum=0.9)
    loss_fn = torch.nn.MSELoss()

    for data, label in CosmoData("/Users/chrisrae/Projects/chris-ml-intern/", batch_size=128):
        opt.zero_grad()
        logits = model.forward(data)
        loss = loss_fn(logits, label)
        loss.backward()
        opt.step()

if __name__ == "__main__":
    main()
