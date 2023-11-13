from pathlib import Path
import sys
path_root = Path(__file__).parents[4]
sys.path.append(str(path_root))

import os

from ML.ResNet50.Torch.data_prep.classes import IMAGENET2012_CLASSES


def main():
    root = "/work/ta127/shared/imagenet-1k/val/"
    with open("/work/ta127/shared/imagenet-1k/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt", "r") as file:
        labels = file.read().split("\n")
    files = os.listdir(root)
    for file in files:
        if ".tar" in file:
            files.remove(file)
    files.sort()
    pairs = zip(labels, files)
    for k in IMAGENET2012_CLASSES.keys():
        try:
            os.mkdir(os.path.join(root, k))
        except FileExistsError:
            pass
        try:
            os.mkdir(os.path.join(root, k, "images"))
        except FileExistsError:
            pass
    
    for pair in pairs:
        os.system(f"cp {os.path.join(root, pair[1])} {os.path.join(root, list(IMAGENET2012_CLASSES.items())[int(pair[0])-1][0], 'images', pair[1])}")


if __name__ == "__main__":
    main()