import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from antorcha.data.loaders import PreprocessedDataLoader
from antorcha.toy_models import CNNParams, BADSettings
from antorcha.toy_models.basic_nn import CNN
from antorcha.train import trainer
from antorcha.utils import fix_ssl_download


def to_gpu(*args):
    return tuple(x.cuda() for x in args)


class Model(nn.Module):
    metric_names = ['Accuracy']

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv_layers = CNN(CNNParams(
            in_channel=3, out_channels=[32, 32, 64, 64], shape=32,
            kernels=[3, 3, 3, 3], strides=[1, 2, 1, 2],
            bad_setting=BADSettings(batchnorm=True, activation=nn.LeakyReLU)
        ))

        self.fc_layers = nn.Sequential(
            nn.Linear(4096, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

    def loss(self, loss, pred, gt):
        return loss(pred, gt)

    def metrics(self, pred: torch.Tensor, gt: torch.Tensor):
        return (pred.argmax(dim=1) == gt).sum().item() / pred.size(0),


if __name__ == '__main__':
    fix_ssl_download.patch()

    train_ds = datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=True, num_workers=1)
    train_loader = PreprocessedDataLoader(train_loader, to_gpu)
    test_loader = PreprocessedDataLoader(test_loader, to_gpu)

    c, h, w = next(iter(train_ds))[0].shape

    model = Model()
    print(model.cuda().train())

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)

    trainer.fit(model, train_loader, test_loader, [loss_fn], opt,
                metrics=True, epochs=30)
