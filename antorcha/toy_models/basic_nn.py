from torch import nn as _nn

from . import util as _util
from ..layers import AutoConv2d as _AConv2d, AutoConvTranspose2d as _AConvT2d, AutoLinear as _ALinear
from ..layers.util import sequential_forward


@sequential_forward
class MLP(_nn.Module):
    def __init__(self, params: _util.MLPParams):
        """
        Construct a multi-layer Fully Connected Neural Network

        The bad settings for the last layer defaults to:
            no BN, `nn.Identity` activation function, no Dropout
        """
        super().__init__()
        i_feat = params.in_feature
        bad = params.bad_setting
        last_bad = params.last_layer_bad or _util.BADSettings()

        self.layers = []
        for i, o_feat in enumerate(params.out_features):
            if i == len(params.out_features) - 1:
                bad = last_bad

            dense = _ALinear(i_feat, o_feat, batchnorm=bad.batchnorm,
                             activation=bad.activation(), dropout=bad.dropout)
            self.add_module(f'linear{i}', dense)
            self.layers.append(dense)
            i_feat = o_feat


@sequential_forward
class CNN(_nn.Module):
    def __init__(self, params: _util.CNNParams):
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


@sequential_forward
class TransposedCNN(_nn.Module):
    def __init__(self, params: _util.CNNParams):
        """
        Construct a multi-layer Transposed Convolutional Neural Network

        The bad settings for the last layer defaults to:
            no BN, `nn.Sigmoid` activation function, no Dropout
        """

        super().__init__()

        ic = params.in_channel
        bad = params.bad_setting
        last_bad = params.last_layer_bad or _util.BADSettings(activation=_nn.Sigmoid)

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


@sequential_forward
class UpSamplingCNN(_nn.Module):
    def __init__(self, params: _util.CNNParams):
        """
        Construct a multi-layer Upsampling Convolutional Neural Network

        The bad settings for the last layer defaults to:
            no BN, `nn.Sigmoid` activation function, no Dropout
        """

        super().__init__()

        self.out_shape = _util.estimate_conv2d_size(
            params.shape, params.strides, params.up_sampling)

        ic = params.in_channel
        bad = params.bad_setting
        last_bad = params.last_layer_bad or _util.BADSettings(activation=_nn.Sigmoid)

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
