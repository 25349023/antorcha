import torch as _torch
from torch import nn as _nn

from . import basic_nn as _basic_nn, util as _util


class Generator(_nn.Module):
    def __init__(self, params: _util.GeneratorParams):
        super().__init__()
        self.in_channel = params.net_params.in_channel
        self.fmap_shape = params.net_params.shape

        self.dense = _nn.Linear(params.z_dim, self.fmap_shape ** 2 * self.in_channel)

        self.generating_network = _basic_nn.UpSamplingCNN(params.net_params)

        self.out_shape = self.generating_network.out_shape

    def forward(self, z):
        x = self.dense(z)
        x = x.reshape((-1, self.in_channel, self.fmap_shape, self.fmap_shape))
        img = self.generating_network(x)
        return img


class Discriminator(_nn.Module):
    def __init__(self, params: _util.CNNParams):
        super().__init__()
        self.in_channel = params.in_channel

        self.discriminating_network = _basic_nn.CNN(params)
        self.fmap_shape = self.discriminating_network.fmap_shape
        self.flatten = _nn.Flatten()
        self.dense = _nn.Linear(self.fmap_shape ** 2 * params.out_channels[-1], 1)
        self.activation = _nn.Sigmoid()

    def forward(self, x):
        x = self.discriminating_network(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.activation(x)
        return x


class GenerativeAdversarialNetwork(_nn.Module):
    loss_names = ['D Loss', 'G Loss']

    def __init__(self, params: _util.GANParams):
        super().__init__()
        self.z_dim = params.gen_params.z_dim

        self.generator = Generator(params.gen_params)
        self.discriminator = Discriminator(params.dis_params)

        self.loss_fn = _nn.BCELoss()

        self.gen_opt = _torch.optim.RMSprop(self.generator.parameters(), lr=params.gen_learning_rate)
        self.dis_opt = _torch.optim.RMSprop(self.discriminator.parameters(), lr=params.dis_learning_rate)

        self.device = 'cpu'

    def forward(self, z):
        fake = self.generator(z)
        pred = self.discriminator(fake)
        return fake, pred

    def generate_images(self, batch_size):
        with _torch.no_grad():
            z = _torch.randn(batch_size, self.z_dim, device='cuda')
            image = self.generator(z).to('cpu').numpy()
            return image

    def forward_dis(self, real_imgs):
        batch_size = real_imgs.shape[0]
        z = _torch.randn(batch_size, self.z_dim, device=self.device)
        ones = _torch.ones((batch_size, 1), device=self.device)
        zeros = _torch.zeros((batch_size, 1), device=self.device)

        fake_imgs = self.generator(z)
        dis_fake_out = self.discriminator(fake_imgs.detach())
        dis_real_out = self.discriminator(real_imgs)

        loss_fake = self.loss_fn(dis_fake_out, zeros)
        loss_real = self.loss_fn(dis_real_out, ones)

        return 0.5 * (loss_fake + loss_real)

    def train_dis(self, real_imgs):
        self.dis_opt.zero_grad()
        loss = self.forward_dis(real_imgs)
        loss.backward()
        self.dis_opt.step()

        return loss.mean().item()

    def forward_gen(self, batch_size):
        z = _torch.randn(batch_size, self.z_dim, device=self.device)
        ones = _torch.ones((batch_size, 1), device=self.device)

        fake_imgs = self.generator(z)
        dis_fake_out = self.discriminator(fake_imgs)
        loss = self.loss_fn(dis_fake_out, ones)

        return loss

    def train_gen(self, batch_size):
        self.gen_opt.zero_grad()
        loss = self.forward_gen(batch_size)
        loss.backward()
        self.gen_opt.step()

        return loss.mean().item()

    def train_adv(self, real_imgs):
        batch_size = real_imgs.shape[0]
        d_loss = self.train_dis(real_imgs)
        g_loss = self.train_gen(batch_size)

        return d_loss, g_loss

    def test_adv(self, real_imgs):
        batch_size = real_imgs.shape[0]
        d_loss = self.forward_dis(real_imgs)
        g_loss = self.forward_gen(batch_size)

        return d_loss, g_loss

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self.to(value)
        self._device = value
