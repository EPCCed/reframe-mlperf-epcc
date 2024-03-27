import h5py
import os
import numpy as np

BASE = "/mnt/ceph_rbd/deepcam/mini/deepcam-data-n512/train"
channels = list(range(16))

def main():
    paths = os.listdir(BASE)
    
    with h5py.File(os.path.join(BASE,"stats.h5"), "r") as f:
        data_shift = f["climate"]["minval"][channels]
        data_scale = 1. / ( f["climate"]["maxval"][channels] - data_shift )
    
    main_f = h5py.File("/mnt/ceph_rbd/deepcam/big_data.h5", "w")
    
    data_dst = main_f.create_dataset("data", (len(paths), 16, 768, 1152), chunks=(1, 16, 768, 1152), dtype="float32")
    label_dst = main_f.create_dataset("labels", (len(paths), 768, 1152),chunks=(1, 768, 1152), dtype="int64")
    
    for i in range(len(paths)):
        path = os.path.join(BASE, paths[i])
        with h5py.File(path, "r") as f:
            d = (f["climate"]["data"][..., channels] - data_shift) * data_scale
            l = f["climate"]["labels_0"]
        
        d = np.transpose(d, (2, 0, 1))
        data_dst[i] = d
        label_dst[i] = l
    
    main_f.close()
        
    
    
if __name__ == "__main__":
    main()