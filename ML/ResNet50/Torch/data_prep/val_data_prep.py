import os 

from tqdm import tqdm

def main():
    base = "/work/ta127/shared/imagenet-1k/data/val"
    base_files = os.listdir(base)
    dirs = list([i for i in base_files if "." not in i])
    files = list([i for i in base_files if "." in i])
    for file in tqdm(files):
        img_class = file.split(".")[0].split("_")[-1]
        if img_class not in dirs:
            os.system(f"mkdir {os.path.join(base, img_class)}")
            dirs.append(img_class)
        os.system(f"mv {os.path.join(base, file)} {os.path.join(base, img_class)}")

if __name__ == "__main__":
    main()