import matplotlib.pyplot as plt
from torch import nn
from torch.distributions.transforms import AffineTransform
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

from antorcha.data.loaders import PreprocessedDataLoader
from antorcha.toy_models import *
from antorcha.train.adversarial_trainer import fit_adv


def to_gpu(*args):
    return args[0].to('cuda', non_blocking=True),


if __name__ == '__main__':
    img_preprocessing = T.Compose([
        T.ToTensor(),
        AffineTransform(loc=-1, scale=2),
    ])

    train_ds = datasets.MNIST(root='../data', train=True, download=True, transform=img_preprocessing)
    test_ds = datasets.MNIST(root='../data', train=False, download=True, transform=img_preprocessing)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=1, prefetch_factor=4)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=True, num_workers=1)

    train_loader = PreprocessedDataLoader(train_loader, to_gpu)
    val_loader = PreprocessedDataLoader(test_loader, to_gpu)
    test_loader = PreprocessedDataLoader(test_loader, to_gpu)

    gp = GeneratorParams(
        net_params=CNNParams(
            in_channel=64,
            out_channels=[128, 128, 64, 1],
            shape=7,
            kernels=[3, 3, 3, 3],
            strides=[1, 1, 1, 1],
            up_sampling=[2, 2, 1, 1],
            bad_setting=BADSettings(batchnorm=True, activation=nn.LeakyReLU),
            last_layer_bad=BADSettings(activation=nn.Tanh),
        ),
        z_dim=100,
        out_shape=(1, 28, 28)
    )

    cp = CNNParams(
        in_channel=1,
        out_channels=[32, 64, 128, 128],
        shape=28,
        kernels=[3, 3, 3, 3],
        strides=[2, 2, 2, 1],
        bad_setting=BADSettings(batchnorm=True, activation=nn.LeakyReLU),
    )

    params = WGANParams(
        gen_params=gp, crtc_params=cp,
        gen_learning_rate=2e-4, crtc_learning_rate=3e-5,
        crtc_weight_threshold=0.008,
        n_critic=5
    )

    wgan = gans.WGAN(params)
    wgan.device = 'cuda'

    fit_adv(wgan, train_loader, test_loader, epochs=20)

    img = wgan.generate_images(80).reshape(80, 28, 28, 1)

    real = next(iter(train_loader))[0]
    real = real.permute(0, 2, 3, 1).cpu().numpy()

    fig, ax = plt.subplots(10, 8, figsize=(20, 25))

    for i in range(10):
        for j in range(8):
            if i % 2 == 0:
                ax[i, j].imshow(img[i * 8 + j], cmap='gray')
            else:
                ax[i, j].imshow(1 - real[(i // 2) * 8 + j], cmap='gray')
    fig.show()

    # torch.save(wgan, 'wgan_minst.pth')
    # torch.save(wgan.state_dict(), 'wgan_minst_weights.pth')
