"""
Data preperation of cosmoflow h5 files to .pt
"""

import os
import click
from functools import partial

import h5py
import torch
from mpi4py import MPI
#from .download import download_cosmoflow


    
def train_test_split(files: list, test_size: float=0.2):
    n = round(int(len(files) * test_size))
    return files[n:], files[:n]

def find_h5_files(dir):
    files = []
    for subdir, _, subfiles in os.walk(dir):
        for f in subfiles:
            if f.endswith('hdf5'):
                files.append(os.path.join(subdir, f))
    return files


def chunk_files_per_worker(files: list, n_workers: int):
    n = len(files) // n_workers 
    int_chunk = [files[x:x+n] for x in range(0, len(files), n)]
    if len(int_chunk) == n_workers + 1:
        remainder = int_chunk.pop()
        for i,r in enumerate(remainder):
            int_chunk[i].append(r)
    return int_chunk


def read_h5(filename):
     with  h5py.File(filename, 'r') as f:
        data = torch.from_numpy(f['full'][:])
        label = torch.from_numpy(f['unitPar'][:])
        return data, label


def chunker(data, size: int=128):
    # change from [512,512,512, 4] -> [4,512,512,512]
    n = data.shape[0] // size
    data = torch.stack(torch.chunk(data, 4, -1),0).squeeze()
    # change from [4,512,512,512] -> [64, 4, 128, 128, 128]
    data = torch.stack(torch.chunk(data, n, 1))
    data = torch.cat(torch.chunk(data, n, 3),0)
    data = torch.cat(torch.chunk(data, n, 4),0)
    return data


def write_pt_files(data, input_file, output_dir):
    output_file = os.path.join(
            output_dir,
            os.path.basename(input_file).replace('.hdf5', '.pt' )
        )
    torch.save(data, output_file)


def process_files(file_names, output_dir):
    with click.progressbar(file_names, label="Processing") as bar:
        for f in bar:
            x, y = read_h5(f)
            x = chunker(x)
            dataset_dict = {"data": x, "label": y}
            write_pt_files(dataset_dict, f, output_dir)


def clean_files(file_names):
    with click.progressbar(file_names, label="Cleaning") as bar:
        for f in bar:
            os.remove(f)


@click.command()
@click.option('--input_dir', type=click.Path(exists=True))
@click.option('--output_dir', type=click.Path())
@click.option("--download", default=False)
@click.option("--clean", default=False)
def main(input_dir, output_dir, download, clean):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if download:
        download_cosmoflow(input_dir)
        comm.Barrier()
        if rank == 0:
            click.echo("Finished Downloading")

    files = find_h5_files(input_dir)
    train_files, test_files = train_test_split(files)
    pw_train = chunk_files_per_worker(train_files, world_size)
    #pw_test = chunk_files_per_worker(test_files, world_size)

    process_files(pw_train[rank], output_dir=os.path.join(output_dir, "train/"))
    #process_files(pw_test[rank], output_dir=output_dir+"/test")
    if clean:
        clean_files(pw_train[rank])
        #clean_files(pw_test[rank])
    comm.Barrier()
    

if __name__ == '__main__':
    main()