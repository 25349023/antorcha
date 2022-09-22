import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import antorcha.toy_models.autoencoders as toy_ae
from antorcha import toy_models as toy
from antorcha.data.datasets import ReconstructionTaskDataset, TransformedDataset
from antorcha.data.loaders import PreprocessedDataLoader
from antorcha.data.transforms import StarCompose
from antorcha.train.trainer import fit
from antorcha.utils import fix_ssl_download
from antorcha.utils.plotting import show_images


def to_gpu(x, y):
    return x.to('cuda'), y.to('cuda')


def random_noise(x, y, std=0.3):
    noised_x = x + torch.randn_like(x) * std
    return noised_x, y


def random_mask(x, y):
    mask = torch.rand_like(x)
    masked_x = x.where(mask > 0.5, torch.zeros_like(x))
    return masked_x, y


if __name__ == '__main__':
    fix_ssl_download.patch()

    train_ds = datasets.MNIST(root='../data', train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.MNIST(root='../data', train=False, download=True, transform=transforms.ToTensor())

    train_ds = TransformedDataset(ReconstructionTaskDataset(train_ds),
                                  StarCompose([random_noise, random_mask]))

    val_ds = TransformedDataset(ReconstructionTaskDataset(test_ds),
                                StarCompose([random_noise, random_mask]))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=1, prefetch_factor=6)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=1, prefetch_factor=6)

    train_loader = PreprocessedDataLoader(train_loader, to_gpu)
    val_loader = PreprocessedDataLoader(val_loader, to_gpu)

    encoder_params = toy.CoderParams(
        net_params=toy.CNNParams(
            in_channel=1, out_channels=[8, 16], shape=28,
            kernels=[3, 3], strides=[2, 2],
            bad_setting=toy.BADSettings(activation=nn.LeakyReLU)
        ),
        z_dim=-1
    )
    decoder_params = toy.symmetric_params(encoder_params)

    auto_encoder = toy_ae.AutoEncoder(encoder_params, decoder_params, with_mlp=False)
    print(auto_encoder.to('cuda').train(), '\n')

    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(auto_encoder.parameters(), lr=5e-4)

    fit(auto_encoder, train_loader, val_loader, [loss_fn], opt, metrics=False, epochs=20)

    # Reconstruction
    # ==============
    auto_encoder.eval()
    img = next(iter(train_loader))[0]

    with torch.no_grad():
        pred = auto_encoder(img)

    show_images(img.cpu())
    show_images(pred.cpu())
