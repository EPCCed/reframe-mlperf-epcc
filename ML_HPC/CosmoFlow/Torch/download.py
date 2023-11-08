import os
import urllib.request 
import shutil
import re
from bs4 import BeautifulSoup
import click
import mpi4py.MPI as MPI


def chunk_files_per_worker(files: list, n_workers: int):
    n = len(files) // n_workers 
    int_chunk = [files[x:x+n] for x in range(0, len(files), n)]
    if len(int_chunk) == n_workers + 1:
        remainder = int_chunk.pop()
        for i,r in enumerate(remainder):
            int_chunk[i].append(r)
    return int_chunk


def download_cosmoflow(download_dir, n_samples=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    if rank == 0:
        base_url = "https://portal.nersc.gov/project/m3363/cosmoUniverse_2019_05_4parE/" 
        dirs = []
        page = urllib.request.urlopen(base_url).read()
        soup = BeautifulSoup(page, 'html.parser')
        for link in soup.findAll('a', attrs={'href': re.compile("^[0-9-_]+/$")}):
            dirs.append(link.get('href'))

        urls = dict()
        for dir in dirs:
            url = base_url + dir
            urls[dir] = []
            page = urllib.request.urlopen(url).read()
            soup = BeautifulSoup(page, 'html.parser')
            for link in soup.findAll('a', attrs={'href': re.compile(".+\.hdf5$")}):
                urls[dir].append(link.get('href'))

            full_urls = []
            for dir in urls.keys:
                for f in urls[dir]:
                    full_urls.append(base_url + dir + f)
        full_urls = list(enumerate(full_urls))[:n_samples]
        pw_urls = chunk_files_per_worker(full_urls, world_size)
        comm.bcast(pw_urls, root=0)

    else:
        pw_urls = comm.bcast(None, root=0)

    def _download_worker(urls, out_dir):
        with click.progressbar(urls, label="Downloading") as bar:
            for i, url in bar:
                out = os.path.join(out_dir, f"{i}.hdf5")
                with urllib.request.urlopen(url) as response, open(out, "wb") as file:
                    shutil.copyfileobj(response, file)

    _download_worker(pw_urls[rank], download_dir)

