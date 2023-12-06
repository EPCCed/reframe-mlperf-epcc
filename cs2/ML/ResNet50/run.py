import os
import sys
sys.path.append("/home/z043/z043/crae-cs1/chris-ml-intern/cs2/modelzoo")

#Append path to parent directory of Cerebras ModelZoo Repos
from modelzoo.common.pytorch.run_utils import run

from data import (
    get_train_dataloader,
    get_val_dataloader,
)
from model import ResNet50Model

def main():
    run(ResNet50Model, get_train_dataloader, get_val_dataloader)

if __name__ == '__main__':
    main()
