import os

from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from ML_HPC.DeepCAM.Torch.data.data_loader import CamDataset, peek_shapes_hdf5

# helper function for determining the data shapes
def get_datashapes(pargs, root_dir):
    
    return peek_shapes_hdf5(os.path.join(root_dir, "train"))
    