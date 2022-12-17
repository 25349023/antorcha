import warnings

import torch as _torch
from torch import nn as _nn

from .basic_nn import _network_selector, _prepare_dense
from .param import CoderParams as _Params, _get_conv_type_from_params
from .util import flatten_length as _flatten_length
from ..train import metric


class Encoder(_nn.Module):
    def __init__(self, params: _Params):
        super().__init__()

        self.encoding_network = _network_selector(params.net_params)
        self.out_shape = self.encoding_network.out_shape

    def forward(self, x):
        x = self.encoding_network(x)
        return x


class Decoder(_nn.Module):
    def __init__(self, params: _Params):
        super().__init__()

        conv_type = _get_conv_type_from_params(params)
        self.decoding_network = _network_selector(params.net_params, conv_type)

    def forward(self, x):
        x = self.decoding_network(x)
        return x


def _gaussian_sampling(mu, log_var):
    epsilon = _torch.normal(0, 1, mu.size(), device=mu.device)
    sigma = _torch.exp(log_var / 2)
    return mu + epsilon * sigma


class VariationalEncoder(_nn.Module):
    def __init__(self, params: _Params):
        super().__init__()

        self.shared_encoder = Encoder(params)
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
    def __init__(self, params: _Params):
        super().__init__()

        self.dense, self.dec_in_shape = _prepare_dense(params.z_dim, params.net_params)
        self.decoding_network = Decoder(params)

    def forward(self, x):
        x = self.dense(x)
        x = x.reshape((-1, *self.dec_in_shape))
        x = self.decoding_network(x)
        return x


class AutoEncoder(_nn.Module):
    def __init__(self, encoder_params: _Params, decoder_params: _Params):
        """
        Vanilla AutoEncoder architecture.

        :param encoder_params: the parameter settings of the encoder, with z_dim ignored
        :param decoder_params: the parameter settings of the encoder
        """
        super().__init__()

        if encoder_params.z_dim != -1:
            warnings.warn('the z_dim parameter of the encoder is ignored in the AutoEncoder, '
                          'set z_dim to -1 to supress this warning', stacklevel=2)

        self.encoder = Encoder(encoder_params)
        self.decoder = Decoder(decoder_params)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def loss(self, r_loss, pred, gt):
        return r_loss(pred, gt)


class VariationalAutoEncoder(_nn.Module):
    def __init__(self, encoder_params: _Params, decoder_params: _Params, r_factor=1000):
        """
        :param encoder_params:
        :param decoder_params:
        """
        super().__init__()

        self.encoder = VariationalEncoder(encoder_params)
        self.decoder = VariationalDecoder(decoder_params)

        self.r_factor = r_factor

        self.metric = metric.KLDivergence(self.encoder)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def loss(self, r_loss, pred, gt):
        r_loss = r_loss(pred, gt)
        kl_div = metric.KLDivergence.kl_loss_fn(self.encoder.mu, self.encoder.log_var)
        return self.r_factor * r_loss + kl_div
