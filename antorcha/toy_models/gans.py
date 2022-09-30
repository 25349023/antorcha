from warnings import warn as _warn

import torch as _torch
from torch import nn as _nn

from . import basic_nn as _basic_nn, util as _util, param as _param
from .basic_nn import _network_selector, _prepare_dense


class Generator(_nn.Module):
    def __init__(self, params: _param.GeneratorParams):
        super().__init__()

        self.dense, self.gen_in_shape = _prepare_dense(params.z_dim, params.net_params)

        conv_type = 'transposed'
        match params.net_params:
            case (_param.CNNParams(up_sampling=u) |
                  [_, _param.CNNParams(up_sampling=u)]) if u is not None:
                conv_type = 'upsampling'

        self.generating_network = _network_selector(params.net_params, conv_type=conv_type)
        self.out_shape = params.out_shape

    def forward(self, z):
        x = self.dense(z)
        x = x.reshape((-1, *self.gen_in_shape))
        img = self.generating_network(x)
        img = img.reshape((-1, *self.out_shape))
        return img


class Discriminator(_nn.Module):
    def __init__(self, params: _param.CNNParams):
        super().__init__()
        self.in_channel = params.in_channel

        mlp_params = _param.MLPParams(in_feature=-1, out_features=[1], bad_setting=_param.BADSettings())
        self.discriminating_network = _basic_nn.CNNWithMLP(params, mlp_params)
        self.activation = _nn.Sigmoid()

    def forward(self, x):
        x = self.discriminating_network(x)
        x = self.activation(x)
        return x


@_util.attach_device_prop
class GAN(_nn.Module):
    loss_names = ['D Loss', 'G Loss']

    def __init__(self, params: _param.GANParams):
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

    def forward_dis(self, real_imgs, fake_imgs):
        batch_size = real_imgs.shape[0]
        ones = _torch.ones((batch_size, 1), device=self.device)
        zeros = _torch.zeros((batch_size, 1), device=self.device)

        dis_fake_out = self.discriminator(fake_imgs)
        dis_real_out = self.discriminator(real_imgs)

        loss_fake = self.loss_fn(dis_fake_out, zeros)
        loss_real = self.loss_fn(dis_real_out, ones)

        return 0.5 * (loss_fake + loss_real)

    def train_dis(self, real_imgs, fake_imgs):
        self.dis_opt.zero_grad()
        loss = self.forward_dis(real_imgs, fake_imgs)
        loss.backward()
        self.dis_opt.step()

        return loss.mean().item()

    def forward_gen(self, fake_imgs):
        batch_size = fake_imgs.shape[0]
        ones = _torch.ones((batch_size, 1), device=self.device)

        dis_fake_out = self.discriminator(fake_imgs)
        loss = self.loss_fn(dis_fake_out, ones)

        return loss

    def train_gen(self, fake_imgs):
        self.gen_opt.zero_grad()
        loss = self.forward_gen(fake_imgs)
        loss.backward()
        self.gen_opt.step()

        return loss.mean().item()

    def train_adv(self, real_imgs):
        batch_size = real_imgs.shape[0]

        z = _torch.randn(batch_size, self.z_dim, device=self.device)
        fake_imgs = self.generator(z)

        # for the sake of efficiency, D and G will share the same fake images,
        # but it requires the fake_imgs fed into D to be detached
        d_loss = self.train_dis(real_imgs, fake_imgs.detach())
        g_loss = self.train_gen(fake_imgs)

        return d_loss, g_loss

    def test_adv(self, real_imgs):
        batch_size = real_imgs.shape[0]

        z = _torch.randn(batch_size, self.z_dim, device=self.device)
        fake_imgs = self.generator(z)

        d_loss = self.forward_dis(real_imgs, fake_imgs)
        g_loss = self.forward_gen(fake_imgs)

        return d_loss, g_loss


class WCritic(_nn.Module):
    def __init__(self, params: _param.CNNParams):
        super().__init__()
        self.in_channel = params.in_channel

        mlp_params = _param.MLPParams(in_feature=-1, out_features=[1], bad_setting=_param.BADSettings())
        self.critic_network = _basic_nn.CNNWithMLP(params, mlp_params)

    def forward(self, x):
        x = self.critic_network(x)
        return x


@_util.attach_device_prop
class WGAN(_nn.Module):
    loss_names = ['C Loss', 'G Loss']

    def __init__(self, params: _param.WGANParams):
        super().__init__()
        self.z_dim = params.gen_params.z_dim

        self.generator = Generator(params.gen_params)
        self.critic = WCritic(params.crtc_params)

        self.gen_opt = _torch.optim.RMSprop(self.generator.parameters(), lr=params.gen_learning_rate)
        self.crtc_opt = _torch.optim.RMSprop(self.critic.parameters(), lr=params.crtc_learning_rate)

        self.n_critic = params.n_critic
        self.crtc_weight_threshold = params.crtc_weight_threshold
        self.steps = 1

        self.device = 'cpu'

    def forward(self, z):
        fake = self.generator(z)
        pred = self.critic(fake)
        return fake, pred

    def generate_images(self, batch_size):
        with _torch.no_grad():
            z = _torch.randn(batch_size, self.z_dim, device='cuda')
            image = self.generator(z).to('cpu').numpy()
            return image

    def forward_crtc(self, real_imgs, fake_imgs):
        crtc_fake_out = self.critic(fake_imgs)
        crtc_real_out = self.critic(real_imgs)

        # maximize real - fake -> minimize fake - real
        loss_fake = _torch.mean(crtc_fake_out)
        loss_real = _torch.mean(crtc_real_out)

        return 0.5 * (loss_fake - loss_real)

    def train_crtc(self, real_imgs, fake_imgs):
        self.crtc_opt.zero_grad()
        loss = self.forward_crtc(real_imgs, fake_imgs)
        loss.backward()
        self.crtc_opt.step()
        self.clip_crtc_weight()

        return loss.mean().item()

    def forward_gen(self, fake_imgs):
        crtc_fake_out = self.critic(fake_imgs)

        # maximize fake -> minimize -fake
        loss = -_torch.mean(crtc_fake_out)
        return loss

    def train_gen(self, fake_imgs):
        self.gen_opt.zero_grad()
        loss = self.forward_gen(fake_imgs)
        loss.backward()
        self.gen_opt.step()

        return loss.mean().item()

    def clip_crtc_weight(self):
        for param in self.critic.parameters():
            param.data.clamp_(-self.crtc_weight_threshold, self.crtc_weight_threshold)

    def train_adv(self, real_imgs):
        batch_size = real_imgs.shape[0]
        z = _torch.randn(batch_size, self.z_dim, device=self.device)
        fake_imgs = self.generator(z)

        c_loss = self.train_crtc(real_imgs, fake_imgs.detach())

        g_loss = None
        if self.steps % self.n_critic == 0:
            self.steps = 1
            g_loss = self.train_gen(fake_imgs)

        self.steps += 1

        return c_loss, g_loss

    def test_adv(self, real_imgs):
        batch_size = real_imgs.shape[0]

        z = _torch.randn(batch_size, self.z_dim, device=self.device)
        fake_imgs = self.generator(z)

        c_loss = self.forward_crtc(real_imgs, fake_imgs)
        g_loss = self.forward_gen(fake_imgs)

        return c_loss, g_loss


@_util.attach_device_prop
class WGANGP(_nn.Module):
    loss_names = ['C Loss', 'G Loss', 'Gradient Penalty']

    def __init__(self, params: _param.WGANGPParams):
        super().__init__()
        self.z_dim = params.gen_params.z_dim

        self.generator = Generator(params.gen_params)

        if params.crtc_params.bad_setting.batchnorm:
            _warn('The critic of WGAN-GP should not have batchnorm layers',
                  RuntimeWarning, stacklevel=2)

        self.critic = WCritic(params.crtc_params)

        self.gen_opt = _torch.optim.Adam(self.generator.parameters(),
                                         lr=params.gen_learning_rate, betas=(0.5, 0.999))
        self.crtc_opt = _torch.optim.Adam(self.critic.parameters(),
                                          lr=params.crtc_learning_rate, betas=(0.5, 0.999))

        self.n_critic = params.n_critic
        self.gp_weight = params.gp_weight
        self.steps = 1

        self.device = 'cpu'

    def forward(self, z):
        fake = self.generator(z)
        pred = self.critic(fake)
        return fake, pred

    def generate_images(self, batch_size):
        with _torch.no_grad():
            z = _torch.randn(batch_size, self.z_dim, device='cuda')
            image = self.generator(z).to('cpu').numpy()
            return image

    def interpolate(self, real_imgs, fake_imgs):
        batch_size = real_imgs.shape[0]

        alpha = _torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolated = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
        return interpolated

    def gradient_penalty(self, pred, interpolated):
        # In order to calculate second order gradient,
        # we need to set create_graph, retain_graph to True
        grads = _torch.autograd.grad(
            pred, interpolated, grad_outputs=_torch.ones_like(pred),
            create_graph=True, retain_graph=True
        )[0]
        l2_norm = _torch.linalg.vector_norm(grads, dim=tuple(range(1, grads.dim())))
        penalty = _torch.square(1 - l2_norm).mean()

        return penalty

    def forward_crtc(self, real_imgs, fake_imgs, train=False):
        crtc_fake_out = self.critic(fake_imgs)
        crtc_real_out = self.critic(real_imgs)

        loss_fake = _torch.mean(crtc_fake_out)
        loss_real = _torch.mean(crtc_real_out)

        loss = loss_fake - loss_real
        penalty = None

        if train:
            inter_imgs = self.interpolate(real_imgs, fake_imgs)
            crtc_inter_out = self.critic(inter_imgs)
            penalty = self.gradient_penalty(crtc_inter_out, inter_imgs)

        return loss, penalty

    def train_crtc(self, real_imgs, fake_imgs):
        self.crtc_opt.zero_grad()
        loss, penalty = self.forward_crtc(real_imgs, fake_imgs, train=True)
        total_loss = loss + self.gp_weight * penalty
        total_loss.backward()
        self.crtc_opt.step()

        return loss.mean().item(), penalty.mean().item()

    def forward_gen(self, fake_imgs):
        crtc_fake_out = self.critic(fake_imgs)
        loss = -_torch.mean(crtc_fake_out)
        return loss

    def train_gen(self, fake_imgs):
        self.gen_opt.zero_grad()
        loss = self.forward_gen(fake_imgs)
        loss.backward()
        self.gen_opt.step()

        return loss.mean().item()

    def train_adv(self, real_imgs):
        batch_size = real_imgs.shape[0]
        z = _torch.randn(batch_size, self.z_dim, device=self.device)
        fake_imgs = self.generator(z)

        c_loss, penalty = self.train_crtc(real_imgs, fake_imgs.detach())

        g_loss = None
        if self.steps % self.n_critic == 0:
            self.steps = 1
            g_loss = self.train_gen(fake_imgs)

        self.steps += 1

        return c_loss, g_loss, penalty

    def test_adv(self, real_imgs):
        batch_size = real_imgs.shape[0]
        z = _torch.randn(batch_size, self.z_dim, device=self.device)
        fake_imgs = self.generator(z)

        c_loss, _ = self.forward_crtc(real_imgs, fake_imgs)
        g_loss = self.forward_gen(fake_imgs)

        return c_loss, g_loss
