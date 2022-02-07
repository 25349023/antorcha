import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

from antorcha.data.loaders import PreprocessedDataLoader
from antorcha.toy_models.alexnet import TinyAlexNet
from antorcha.train import trainer
from antorcha.utils import fix_ssl_download


def to_gpu(*args):
    return tuple(x.to('cuda', non_blocking=True) for x in args)


class StandardizeTransform:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def __call__(self, img):
        return (img - self.mean) / self.std

    def set_dist(self, mean, std):
        self.mean = mean.transpose((2, 0, 1))
        self.std = std.transpose((2, 0, 1))


if __name__ == '__main__':
    fix_ssl_download.patch()
    torch.backends.cudnn.benchmark = True

    std_transform = StandardizeTransform()

    img_preprocessing = T.Compose([
        T.ToTensor(),
        std_transform,
        T.Pad(4),
        T.RandomCrop(32),
        T.RandomHorizontalFlip(0.5),
    ])

    test_preprocessing = T.Compose([
        T.ToTensor(),
        std_transform
    ])

    train_ds = datasets.CIFAR10(root='../data', train=True, download=True, transform=img_preprocessing)
    test_ds = datasets.CIFAR10(root='../data', train=False, download=True, transform=test_preprocessing)

    mean = train_ds.data.mean(axis=0, dtype=np.float32) / 255
    std = train_ds.data.std(axis=0, dtype=np.float32) / 255
    std_transform.set_dist(mean, std)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, prefetch_factor=4)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=True, num_workers=1)
    train_loader = PreprocessedDataLoader(train_loader, to_gpu)
    test_loader = PreprocessedDataLoader(test_loader, to_gpu)

    c, h, w = next(iter(train_ds))[0].shape

    model = TinyAlexNet(32).to('cuda', non_blocking=True)

    print(model)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    sch = torch.optim.lr_scheduler.StepLR(opt, 6, 0.45)

    trainer.fit(model, train_loader, test_loader, [loss_fn], opt,
                metrics=True, scheduler=sch, epochs=40)
