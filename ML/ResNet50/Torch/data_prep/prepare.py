from PIL import Image
from mpi4py import MPI
import click
import os
import time

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def chunk_files_per_worker(files: list, n_workers: int):
    return [files[i::n_workers] for i in range(n_workers)]
    n = len(files) // n_workers
    int_chunk = [files[x:x+n] for x in range(0, len(files), n)]
    if len(int_chunk) == n_workers + 1:
        remainder = int_chunk.pop()
        for i,r in enumerate(remainder):
            int_chunk[i].append(r)
    return int_chunk

def extract_tar(comm: MPI.Intracomm, clean: bool):
    paths = ["train/ILSVRC2012_img_train.tar"]#, "val/ILSVRC2012_img_val.tar", "test/ILSVRC2012_img_test_v10102019.tar"]
    base = "/work/ta127/shared/imagenet-1k/"
    if comm.Get_rank() == 0:
        for path in paths:
            print("Started Uncompressing Downloads")
            os.system(f"tar -xf {os.path.join(base, path)} -C {os.path.join(base, path.split('/')[0])}")
            if clean:
                os.remove(os.path.join(base, path))
        print("Finished Uncompressing Downloads")
    comm.Barrier()


@click.command()
@click.option("--clean", type=bool, default=False)
def main(clean):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    base = "/work/ta127/shared/imagenet-1k/"

    if len(os.listdir("/work/ta127/shared/imagenet-1k/train/")) == 1:
        extract_tar(comm, clean)
    
    dirs = ["train/"]#, "test/", "val/"]
    for dir in dirs:
        if rank == 0:
            print(f"Started Preprocessing {dir}")
        files = os.listdir(os.path.join(base, dir))
        files.sort()
        for f in files:
            if "ILSVRC2012" in f:
                files.remove(f)

        pw_files = chunk_files_per_worker(files, world_size)
        for f in pw_files[rank]:
            print(f"    Started Preprocessing {f}")
            full_path = os.path.join(base, dir, f)
            output_path = os.path.join(full_path.replace(".tar", "/"), "images/")
            try:
                os.mkdir(full_path.replace(".tar", ""))
            except FileExistsError:
                pass
            try:
                os.mkdir(output_path)
            except FileExistsError:
                pass
            if len(os.listdir(output_path)) == 0:
                os.system(f"tar -xf {full_path} -C {output_path}")
                time.sleep(20)
            if clean and full_path.split("/")[-1] in os.listdir("/".join(full_path.split("/")[:-1])):
                try:
                    if ".tar" not in full_path:
                        os.remove(full_path + ".tar")
                    else:
                        os.remove(full_path)
                except FileNotFoundError:
                    pass
            continue
            for img in os.listdir(output_path):
                img_path = os.path.join(output_path, img)
                img = Image.open(img_path)
                rgb_img = img.convert("RGB")
                if ".JPEG" in img_path:
                    pass
                else:
                    if ".png":
                        os.remove(img_path)
                        img_path=img_path.replace(".png", ".JPEG")
                    elif ".PNG":
                        os.remove(img_path)
                        img_path=img_path.replace(".PNG", ".JPEG")
                rgb_img.save(img_path, )

if __name__ == "__main__":
    main()