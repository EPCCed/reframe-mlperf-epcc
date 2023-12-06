from mpi4py import MPI
import os

def chunk_files_per_worker(files: list, n_workers: int):
    n = len(files) // n_workers 
    int_chunk = [files[x:x+n] for x in range(0, len(files), n)]
    if len(int_chunk) == n_workers + 1:
        remainder = int_chunk.pop()
        for i,r in enumerate(remainder):
            int_chunk[i].append(r)
    return int_chunk

def download_imagenet():
    urls = ["https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar",
            "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
            "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar",
    ]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    pw_urls = chunk_files_per_worker(urls, world_size)
    for url in pw_urls[rank]:
        file_name = url.replace("https://image-net.org/data/ILSVRC/2012/", "")
        base = "/work/ta127/shared/imagenet-1k/"    
        if "train" in url:
            data_type = "train/"
        elif "val" in url:
            data_type = "val/"
        elif "test" in url:
            data_type = "test/"
        try:
            print(f"curl -O {os.path.join(base, data_type, file_name)} {url}")
            #os.system(f"curl -O {os.path.join(base, data_type, file_name)} {url}")
        except Exception as e:
            with open("/work/ta127/shared/imagenet-1k/errors.txt", "a") as f:
                f.write(str(e))
            exit()

if __name__ == "__main__":
    download_imagenet()
