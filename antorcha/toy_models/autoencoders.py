from typing import Union as _Union

import torch as _torch
from torch import nn as _nn

from .basic_nn import (
    CNN as _CNN,
    TransposedCNN as _TransposedCNN,
    UpSamplingCNN as _UpSamplingCNN,
    MLP as _MLP
)
from .util import CoderParams as _Params, CNNParams as _CNNParams, MLPParams as _MLPParams


def _network_selector(params: _Union[_CNNParams, _MLPParams], conv_type='normal'):
    if isinstance(params, _CNNParams):
        if conv_type == 'normal':
            return _CNN(params)
        elif conv_type == 'transposed':
            return _TransposedCNN(params)
        elif conv_type == 'upsampling':
            return _UpSamplingCNN(params)
    elif isinstance(params, _MLPParams):
        return _MLP(params)


class Encoder(_nn.Module):
    def __init__(self, params: _Params):
        super().__init__()
        self.encoding_network = _network_selector(params.net_params)
        self.fmap_shape = self.encoding_network.fmap_shape
        self.flatten = _nn.Flatten()
        self.dense = _nn.Linear(
            self.fmap_shape ** 2 * params.net_params.out_channels[-1], params.z_dim)

    def forward(self, x):
        x = self.encoding_network(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


def _gaussian_sampling(mu, log_var):
    epsilon = _torch.normal(0, 1, mu.size(), device=mu.device)
    sigma = _torch.exp(log_var / 2)
    return mu + epsilon * sigma


class VariationalEncoder(_nn.Module):
    def __init__(self, params: _Params):
        super().__init__()
        self.encoding_network = _network_selector(params.net_params)
        self.fmap_shape = self.encoding_network.fmap_shape

        self.flatten = _nn.Flatten()
        self.dense_mu = _nn.Linear(
            self.fmap_shape ** 2 * params.net_params.out_channels[-1], params.z_dim)
        self.dense_log_var = _nn.Linear(
            self.fmap_shape ** 2 * params.net_params.out_channels[-1], params.z_dim)

        self.mu = None
        self.log_var = None

    def forward(self, x):
        x = self.encoding_network(x)
        x = self.flatten(x)
        self.mu = self.dense_mu(x)
        self.log_var = self.dense_log_var(x)
        y = _gaussian_sampling(self.mu, self.log_var)
        return y


class Decoder(_nn.Module):
    def __init__(self, params: _Params):
        super().__init__()
        net_p = params.net_params
        self.in_channel = net_p.in_channel
        self.fmap_shape = net_p.shape
        self.dense = _nn.Linear(params.z_dim, net_p.shape ** 2 * net_p.in_channel)
        self.decoding_network = _network_selector(params.net_params, conv_type='transposed')

    def forward(self, x):
        x = self.dense(x)
        x = x.reshape((-1, self.in_channel, self.fmap_shape, self.fmap_shape))
        x = self.decoding_network(x)
        return x


class AutoEncoder(_nn.Module):
    def __init__(self, encoder_params: _Params, decoder_params: _Params, auto_shape=False):
        """
        :param encoder_params:
        :param decoder_params:
        :param auto_shape:
            automatically computes the value of decoder input fmap shape (conv only)
        """
        super().__init__()

        self.encoder = Encoder(encoder_params)
        if auto_shape and isinstance(decoder_params.net_params, _CNNParams):
            decoder_params._replace(
                net_params=decoder_params.net_params._replace(
                    shape=self.encoder.fmap_shape)
            )
        self.decoder = Decoder(decoder_params)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def loss(self, r_loss, pred, gt):
        return r_loss(pred, gt)


class VariationalAutoEncoder(_nn.Module):
    def __init__(self, encoder_params: _Params, decoder_params: _Params, auto_shape=False, r_factor=1000):
        """
        :param encoder_params:
        :param decoder_params:
        :param auto_shape: automatically computes the value of decoder input fmap shape
        """
        super().__init__()

        self.encoder = VariationalEncoder(encoder_params)
        if auto_shape and isinstance(decoder_params.net_params, _CNNParams):
            decoder_params._replace(
                net_params=decoder_params.net_params._replace(
                    shape=self.encoder.fmap_shape)
            )
        self.decoder = Decoder(decoder_params)

        self.r_factor = r_factor
        self.r_loss = _torch.tensor(0.0)
        self.kl_div = _torch.tensor(0.0)

        self.metric_names = ['Reconstruct Loss', 'KL Divergence']

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    @staticmethod
    def kl_loss_fn(mu, log_var):
        kl = 0.5 * _torch.sum(_torch.square(mu) + _torch.exp(log_var) - 1 - log_var, dim=1)
        return _torch.mean(kl)

    def loss(self, r_loss, pred, gt):
        self.r_loss = r_loss(pred, gt)
        self.kl_div = self.kl_loss_fn(self.encoder.mu, self.encoder.log_var)
        return self.r_factor * self.r_loss + self.kl_div

    def metrics(self, pred, gt):
        return [self.r_loss.item(), self.kl_div.item()]
