import functools
import operator

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from antorcha.data.loaders import PreprocessedDataLoader
from antorcha.train import trainer, metric
from antorcha.utils import fix_ssl_download


def to_gpu(*args):
    return tuple(x.cuda() for x in args)


class Model(nn.Module):
    def __init__(self, *input_dims):
        super().__init__()
        self.input_units = functools.reduce(operator.mul, input_dims)
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(self.input_units, 200),
            nn.ReLU(),
            nn.Linear(200, 150),
            nn.ReLU(),
            nn.Linear(150, 10)
        )

        self.metric = metric.Accuracy()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

    def loss(self, loss, pred, gt):
        return loss(pred, gt)


if __name__ == '__main__':
    fix_ssl_download.patch()

    train_ds = datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=True, num_workers=1)
    train_loader = PreprocessedDataLoader(train_loader, to_gpu)
    test_loader = PreprocessedDataLoader(test_loader, to_gpu)

    c, h, w = next(iter(train_ds))[0].shape

    model = Model(c, h, w)
    print(model.cuda().train())

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)

    trainer.fit(model, train_loader, test_loader, [loss_fn], opt,
                metrics=True, epochs=30)
