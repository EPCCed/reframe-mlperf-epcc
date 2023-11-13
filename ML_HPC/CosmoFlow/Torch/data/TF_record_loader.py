import torch
import torchdata

def get_datapipe(path):
    datapipe1 = torchdata.datapipes.iter.FileLister(path, "*.tfrecord")
    datapipe2 = torchdata.datapipes.iter.FileOpener(datapipe1, mode="b")
    return datapipe2.load_from_tfrecord()
from tfrecord.torch.dataset import TFRecordDataset
import os
from torchdata.datapipes.iter import FileLister, FileOpener, TFRecordLoader

if __name__ == "__main__":
    """
    path = "/work/z19/shared/mlperf-hpc/cosmoflow/full/cosmoUniverse_2019_05_4parE_tf_v2/train/"
    file = os.listdir(path)[0]
    full_path = os.path.join(path, file)
    print(full_path)
    desc = {"x": "str", "y": "float"}
    dataset = TFRecordDataset(full_path, None, desc)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    data = next(iter(loader))
    print(data)
    """
    path = "/work/z19/shared/mlperf-hpc/cosmoflow/full/cosmoUniverse_2019_05_4parE_tf_v2/train/"
    datapipe1 = FileLister(path, "22098324_univ_ics_2019-03_a1099062_*.tfrecord")
    print(len(list(datapipe1)))
    datapipe2 = FileOpener(datapipe1, mode="b")
    tfrecord_loader_dp = TFRecordLoader(datapipe2)
    print(dir(tfrecord_loader_dp))
    for i in list(tfrecord_loader_dp):
        print(i)
        exit()
