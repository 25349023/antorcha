import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import antorcha.toy_models.autoencoders as toy_ae
from antorcha import toy_models as toy
from antorcha.data.loaders import PreprocessedDataLoader
from antorcha.train.trainer import fit
from antorcha.utils import fix_ssl_download
from antorcha.utils.plotting import visualize_latent_space_dist, plot_interpolation, show_images


def to_gpu(x, y):
    return x.to('cuda'), y.to('cuda')


def reconstruct_task(x, y):
    return x.to('cuda'), x.clone().to('cuda')


if __name__ == '__main__':
    fix_ssl_download.patch()

    train_ds = datasets.MNIST(root='../data', train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.MNIST(root='../data', train=False, download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, prefetch_factor=6)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=True, num_workers=2, prefetch_factor=6)

    train_loader = PreprocessedDataLoader(train_loader, reconstruct_task)
    val_loader = PreprocessedDataLoader(test_loader, reconstruct_task)
    test_loader = PreprocessedDataLoader(test_loader, to_gpu)

    encoder_params = toy.CoderParams(
        net_params=(
            toy.CNNParams(
                in_channel=1, out_channels=[32, 64, 64, 64], shape=28,
                kernels=[3, 3, 3, 3], strides=[1, 2, 2, 1],
                bad_setting=toy.BADSettings(activation=nn.LeakyReLU)
            ),
            toy.MLPParams(
                in_feature=-1, out_features=[16], bad_setting=toy.BADSettings()
            )
        ),
        z_dim=2
    )

    decoder_params = toy.symmetric_params(encoder_params)

    vae = toy_ae.VariationalAutoEncoder(encoder_params, decoder_params, r_factor=800)
    print(vae.to('cuda').train(), '\n')

    r_loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(vae.parameters(), lr=5e-4)

    fit(vae, train_loader, val_loader, [r_loss_fn], opt, metrics=True, epochs=20)

    # Reconstruction
    # ==============
    vae.eval()
    img = next(iter(train_loader))[0]

    with torch.no_grad():
        pred = vae(img)

    show_images(img.cpu())
    show_images(pred.cpu())

    # Dist. of Latent space
    # =====================
    visualize_latent_space_dist(vae.encoder, test_loader, 'VAE')
    plot_interpolation(vae.decoder, 2, 28)
