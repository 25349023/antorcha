import numpy as _np
from torch.utils import data as _data


# [TODO] Generalize this to Doodle Dataset
class CamelDataset(_data.Dataset):
    def __init__(self, transform=None, train=True):
        self.data = _np.load('data/camel/full_numpy_bitmap_camel.npy')
        self.data = self.data.reshape((-1, 28, 28, 1))
        if train:
            self.data = self.data[:-10000]
        else:
            self.data = self.data[-10000:]
        self.transform = transform

    def __getitem__(self, item):
        image = self.data[item]
        if self.transform:
            image = self.transform(image)
        return image,

    def __len__(self):
        return self.data.shape[0]
