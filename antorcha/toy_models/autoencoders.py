import warnings

import torch as _torch
from torch import nn as _nn

from .basic_nn import _network_selector
from .param import CNNParams as _CNNParams, CoderParams as _Params, MLPParams as _MLPParams
from .util import flatten_length as _flatten_length


class Encoder(_nn.Module):
    def __init__(self, params: _Params, with_mlp=True):
        super().__init__()

        # since isinstance(params, tuple) does not work,
        # we check if it is an ordinary tuple by testing its length
        if with_mlp and len(params.net_params) != 2:
            raise ValueError('CNNWithMLP requires two sets of network params, but only 1 found')

        self.encoding_network = _network_selector(params.net_params)
        self.out_shape = self.encoding_network.out_shape

        if params.z_dim not in (-1, self.out_shape):
            warnings.warn('The actual output dimension does not match '
                          'the z_dim specified by the CoderParams.',
                          RuntimeWarning, stacklevel=2)

    def forward(self, x):
        x = self.encoding_network(x)
        return x


class Decoder(_nn.Module):
    def __init__(self, params: _Params, with_mlp=True):
        super().__init__()

        # since isinstance(params, tuple) does not work,
        # we check if it is an ordinary tuple by testing its length
        if with_mlp and len(params.net_params) != 2:
            raise ValueError('CNNWithMLP requires two sets of network params, but only one found')

        # [TODO] add support to upsampling Decoder
        self.decoding_network = _network_selector(params.net_params, conv_type='transposed')

    def forward(self, x):
        x = self.decoding_network(x)
        return x


def _gaussian_sampling(mu, log_var):
    epsilon = _torch.normal(0, 1, mu.size(), device=mu.device)
    sigma = _torch.exp(log_var / 2)
    return mu + epsilon * sigma


class VariationalEncoder(_nn.Module):
    def __init__(self, params: _Params, with_mlp=True):
        super().__init__()

        self.shared_encoder = Encoder(params, with_mlp)
        self.flatten = None
        if len(self.shared_encoder.out_shape) > 1:
            self.flatten = _nn.Flatten()

        in_feat = _flatten_length(self.shared_encoder.out_shape)
        self.dense_mu = _nn.Linear(in_feat, params.z_dim)
        self.dense_log_var = _nn.Linear(in_feat, params.z_dim)

        self.mu = None
        self.log_var = None

    def forward(self, x):
        x = self.shared_encoder(x)
        if self.flatten:
            x = self.flatten(x)
        self.mu = self.dense_mu(x)
        self.log_var = self.dense_log_var(x)
        y = _gaussian_sampling(self.mu, self.log_var)
        return y


class VariationalDecoder(_nn.Module):
    def __init__(self, params: _Params, with_mlp=True):
        super().__init__()

        self.fmap_shape = ()
        match params.net_params:
            case (_MLPParams() as p) | (_MLPParams() as p, _):
                self.dense = _nn.Linear(params.z_dim, p.in_feature)
            case _CNNParams() as p:
                self.dense = _nn.Linear(params.z_dim, p.shape ** 2 * p.in_channel)
                self.fmap_shape = (p.in_channel, p.shape, p.shape)

        self.decoding_network = Decoder(params, with_mlp=with_mlp)

    def forward(self, x):
        x = self.dense(x)
        if self.fmap_shape:
            x = x.reshape((-1, *self.fmap_shape))
        x = self.decoding_network(x)
        return x


class AutoEncoder(_nn.Module):
    def __init__(self, encoder_params: _Params, decoder_params: _Params, with_mlp=True):
        """
        :param encoder_params:
        :param decoder_params:
        """
        super().__init__()

        self.encoder = Encoder(encoder_params, with_mlp=with_mlp)
        self.decoder = Decoder(decoder_params, with_mlp=with_mlp)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def loss(self, r_loss, pred, gt):
        return r_loss(pred, gt)


class VariationalAutoEncoder(_nn.Module):
    metric_names = ['Reconstruct Loss', 'KL Divergence']

    def __init__(self, encoder_params: _Params, decoder_params: _Params, r_factor=1000, with_mlp=True):
        """
        :param encoder_params:
        :param decoder_params:
        """
        super().__init__()

        self.encoder = VariationalEncoder(encoder_params, with_mlp)
        self.decoder = VariationalDecoder(decoder_params, with_mlp)

        self.r_factor = r_factor
        self.r_loss = _torch.tensor(0.0)
        self.kl_div = _torch.tensor(0.0)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    @staticmethod
    def kl_loss_fn(mu, log_var):
        kl = 0.5 * _torch.sum(mu ** 2 + log_var.exp() - 1 - log_var, dim=1)
        return _torch.mean(kl)

    def loss(self, r_loss, pred, gt):
        self.r_loss = r_loss(pred, gt)
        self.kl_div = self.kl_loss_fn(self.encoder.mu, self.encoder.log_var)
        return self.r_factor * self.r_loss + self.kl_div

    def metrics(self, pred, gt):
        return [self.r_loss.item(), self.kl_div.item()]
