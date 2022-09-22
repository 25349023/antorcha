import os.path as _pth

import numpy as _np
from torch.utils import data as _data


# [TODO] Generalize this to Doodle Dataset
class CamelDataset(_data.Dataset):
    def __init__(self, path_to_data_root='.', transform=None, train=True):
        self.data = _np.load(_pth.join(path_to_data_root, 'data', 'camel', 'full_numpy_bitmap_camel.npy'))
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


class ReconstructionTaskDataset(_data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        x, y = self.dataset[item]
        return x, x.clone()

    def __len__(self):
        return len(self.dataset)


class TransformedDataset(_data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        data = self.dataset[item]
        try:
            transformed_data = self.transform(data)
        except TypeError:
            print(self.transform, len(data))
            raise
        return transformed_data

    def __len__(self):
        return len(self.dataset)
