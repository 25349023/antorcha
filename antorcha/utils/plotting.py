import numpy as _np
import torch as _torch
from matplotlib import pyplot as _plt


def visualize_latent_space_dist(encoder, data_loader, name):
    points = []
    color = []

    with _torch.no_grad():
        for x, y in data_loader:
            embedded: _torch.Tensor = encoder(x)
            points.extend(embedded.to('cpu').numpy())
            color.extend(y.to('cpu').numpy() / 10 + 0.05)

    points = _np.asarray(points)
    color = _np.asarray(color)

    fig, ax = _plt.subplots(1, 1, figsize=(14, 14))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
    ax.scatter(points[:, 0], points[:, 1], c=color)
    ax.set_title(f'Latent Space of {name}')
    fig.show()


def plot_interpolation(decoder, z_dim, img_size, num_divide=20):
    z1 = _torch.randn([1, z_dim])
    z2 = _torch.randn([1, z_dim])
    diff = (z2 - z1) / (num_divide - 1)
    zs = [z1 + i * diff for i in range(num_divide)]

    fig, ax = _plt.subplots(1, num_divide, sharey=True, figsize=(num_divide + 5, 2))
    fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98)

    with _torch.no_grad():
        for i, z in enumerate(zs):
            z = z.cuda()
            pred = decoder(z)
            ax[i].imshow(pred.reshape(-1, img_size, img_size).permute(1, 2, 0).cpu().numpy(), cmap='gray')
    fig.show()
