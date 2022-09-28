import warnings

from torch import nn as _nn

from . import BasicNNParams as _BasicNNParams, MLPParams as _MLPParams
from ..layers import AutoConv2d as _AConv2d, AutoConvTranspose2d as _AConvT2d, AutoLinear as _ALinear
from ..layers.util import sequential_forward as _seq_forward
from ..toy_models import util as _util, param as _param
from ..utils.aux_types import maybe_pair as _maybe_pair


@_seq_forward
class MLP(_nn.Module):
    def __init__(self, params: _param.MLPParams):
        """
        Construct a multi-layer Fully Connected Neural Network

        The bad settings for the last layer defaults to:
            no BN, `nn.Identity` activation function, no Dropout
        """
        super().__init__()
        i_feat = params.in_feature
        bad = params.bad_setting
        last_bad = params.last_layer_bad or _param.BADSettings()

        if not params.out_features:
            raise ValueError("number of layers should not be zero")

        self.layers = []
        for i, o_feat in enumerate(params.out_features):
            if i == len(params.out_features) - 1:
                bad = last_bad

            dense = _ALinear(i_feat, o_feat, batchnorm=bad.batchnorm,
                             activation=bad.activation(), dropout=bad.dropout)
            self.add_module(f'linear{i}', dense)
            self.layers.append(dense)
            i_feat = o_feat

        self.out_shape = (params.out_features[-1],)


@_seq_forward
class CNN(_nn.Module):
    def __init__(self, params: _param.CNNParams):
        """
        Construct a multi-layer Convolutional Neural Network

        The bad settings for the last layer defaults to be the same as previous layers
        """
        super().__init__()
        self.fmap_shape = _util.estimate_conv2d_size(params.shape, params.strides)

        ic = params.in_channel
        bad = params.bad_setting
        last_bad = params.last_layer_bad or bad

        self.layers = []
        conv_args = zip(params.out_channels, params.kernels, params.strides)
        for i, (oc, k, s) in enumerate(conv_args):
            if i == len(params.out_channels) - 1:
                bad = last_bad

            conv = _AConv2d(ic, oc, k, s, batchnorm=bad.batchnorm,
                            activation=bad.activation(), dropout=bad.dropout)
            self.add_module(f'conv{i}', conv)
            self.layers.append(conv)
            ic = oc

        self.out_shape = (params.out_channels[-1], self.fmap_shape, self.fmap_shape)


@_seq_forward
class TransposedCNN(_nn.Module):
    def __init__(self, params: _param.CNNParams):
        """
        Construct a multi-layer Transposed Convolutional Neural Network

        The bad settings for the last layer defaults to:
            no BN, `nn.Sigmoid` activation function, no Dropout
        """

        super().__init__()

        ic = params.in_channel
        bad = params.bad_setting
        last_bad = params.last_layer_bad or _param.BADSettings(activation=_nn.Sigmoid)

        self.layers = []
        conv_args = zip(params.out_channels, params.kernels, params.strides)
        for i, (oc, k, s) in enumerate(conv_args):
            if i == len(params.out_channels) - 1:
                bad = last_bad

            conv_t = _AConvT2d(ic, oc, k, s, batchnorm=bad.batchnorm,
                               activation=bad.activation(), dropout=bad.dropout)

            self.add_module(f'conv_t{i}', conv_t)
            self.layers.append(conv_t)
            ic = oc

        s = _util.estimate_conv2d_size(params.shape, up_sample=params.strides)
        self.out_shape = [params.out_channels[-1], s, s]


@_seq_forward
class UpSamplingCNN(_nn.Module):
    def __init__(self, params: _param.CNNParams):
        """
        Construct a multi-layer Upsampling Convolutional Neural Network

        The bad settings for the last layer defaults to:
            no BN, `nn.Sigmoid` activation function, no Dropout
        """

        super().__init__()

        ic = params.in_channel
        bad = params.bad_setting
        last_bad = params.last_layer_bad or _param.BADSettings(activation=_nn.Sigmoid)

        self.layers = []
        conv_args = zip(params.out_channels, params.kernels, params.strides, params.up_sampling)
        for i, (oc, k, s, u) in enumerate(conv_args):
            if i == len(params.out_channels) - 1:
                bad = last_bad

            if u > 1:
                up_s = _nn.Upsample(scale_factor=u, mode='nearest')
                self.add_module(f'up_sample{i}', up_s)
                self.layers.append(up_s)

            conv = _AConv2d(ic, oc, k, s, batchnorm=bad.batchnorm,
                            activation=bad.activation(), dropout=bad.dropout)
            self.add_module(f'conv{i}', conv)
            self.layers.append(conv)
            ic = oc

        s = _util.estimate_conv2d_size(params.shape, params.strides, params.up_sampling)
        self.out_shape = (s, s, params.out_channels[-1])


@_seq_forward
class CNNWithMLP(_nn.Module):
    def __init__(self, cnn_params: _param.CNNParams, mlp_params: _param.MLPParams):
        """
        Construct a network that contains CNN and MLP

        The bad settings for the last layer default to those in CNN and MLP respectively
        """

        super().__init__()
        self.in_channel = cnn_params.in_channel
        self.cnn = CNN(cnn_params)
        self.fmap_shape = self.cnn.fmap_shape

        self.flatten = _nn.Flatten()
        mlp_in_feat = self.fmap_shape ** 2 * cnn_params.out_channels[-1]

        if mlp_params.in_feature not in (-1, mlp_in_feat):
            warnings.warn('MLP in_feature does not match the output feature of cnn, '
                          'and this will be corrected automatically.',
                          RuntimeWarning, stacklevel=2)

        mlp_params = mlp_params._replace(in_feature=mlp_in_feat)
        self.mlp = MLP(mlp_params)

        self.layers = [self.cnn, self.flatten, self.mlp]
        self.out_shape = (mlp_params.out_features[-1],)


class MLPWithTCNN(_nn.Module):
    def __init__(self, mlp_params: _param.MLPParams, tcnn_params: _param.CNNParams):
        """
        Construct a network that contains MLP and Transposed CNN

        The bad settings for the last layer default to
        those in MLP and TransposedCNN respectively
        """

        super().__init__()

        self.mlp = MLP(mlp_params)
        self.tcnn_in_shape = (tcnn_params.in_channel,) + (tcnn_params.shape,) * 2
        self.tcnn = TransposedCNN(tcnn_params)

        s = _util.estimate_conv2d_size(tcnn_params.shape, up_sample=tcnn_params.strides)
        self.out_shape = (tcnn_params.out_channels[-1], s, s)

    def forward(self, x):
        x = self.mlp(x)
        x = x.reshape((-1, *self.tcnn_in_shape))
        x = self.tcnn(x)
        return x


def _network_selector(params: _maybe_pair[_BasicNNParams], conv_type='normal'):
    if isinstance(params, _MLPParams):
        return MLP(params)

    if conv_type == 'normal':
        return CNN(params)
    elif conv_type == 'cnn_mlp':
        return CNNWithMLP(*params)
    elif conv_type == 'transposed':
        return TransposedCNN(params)
    elif conv_type == 'mlp_transposed':
        return MLPWithTCNN(*params)
    elif conv_type == 'upsampling':
        return UpSamplingCNN(params)
