import os
import sys

#Append path to parent directory of Cerebras ModelZoo Repository
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from modelzoo.common.pytorch.run_utils import run

from data import (
    get_train_dataloader,
    get_val_dataloader,
)
from model import CosmoFlowModel

def main():
    run(CosmoFlowModel, get_train_dataloader, get_val_dataloader)

if __name__ == '__main__':
    main()