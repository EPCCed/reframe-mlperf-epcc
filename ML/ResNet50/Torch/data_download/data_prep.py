import os 
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm

def main(files, base):
    base_files = os.listdir(base)
    dirs = list([i for i in base_files if "." not in i])
    del base_files
    for file in tqdm(files):
        img_class = file.split("_")[0]
        if img_class not in dirs:
            os.system(f"mkdir {os.path.join(base, img_class)}")
            dirs.append(img_class)
        os.system(f"mv {os.path.join(base, file)} {os.path.join(base, img_class)}")

if __name__ == "__main__":
    base = "/home/eidf095/eidf095/crae-ml/imagenet-1k/data/train"
    base_files = os.listdir(base)
    base_files.sort()
    n = 8
    files = list([i for i in base_files if "." in i])
    files = list([files[i::n] for i in range(n)])
    p = Pool(n)
    p.map(partial(main, base=base), files)
