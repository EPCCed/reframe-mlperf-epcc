import torch
import torchdata

def get_datapipe(path):
    datapipe1 = torchdata.datapipes.iter.FileLister(path, "*.tfrecord")
    datapipe2 = torchdata.datapipes.iter.FileOpener(datapipe1, mode="b")
    return datapipe2.load_from_tfrecord()
