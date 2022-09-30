from typing import NamedTuple as _NamedTuple, Callable as _Callable, Union as _Union

from torch import nn as _nn

from . import util as _util
from ..utils.aux_types import maybe_pair as _maybe_pair


class BADSettings(_NamedTuple):
    batchnorm: bool = False
    activation: _Callable = _nn.Identity
    dropout: float = 0.0


class MLPParams(_NamedTuple):
    in_feature: int
    out_features: list[int]
    bad_setting: BADSettings
    last_layer_bad: BADSettings = None


class CNNParams(_NamedTuple):
    in_channel: int
    out_channels: list[int]
    shape: int
    kernels: list[int]
    strides: list[int]
    up_sampling: list[int] = None
    bad_setting: BADSettings = BADSettings()
    last_layer_bad: BADSettings = None


_SimpleNNParams = _Union[MLPParams, CNNParams]
BasicNNParams = _maybe_pair[_SimpleNNParams]


class CoderParams(_NamedTuple):
    net_params: BasicNNParams
    z_dim: int


class GeneratorParams(_NamedTuple):
    net_params: BasicNNParams
    z_dim: int
    out_shape: tuple[int, ...]


class GANParams(_NamedTuple):
    gen_params: GeneratorParams
    dis_params: CNNParams
    gen_learning_rate: float
    dis_learning_rate: float


class WGANParams(_NamedTuple):
    gen_params: GeneratorParams
    crtc_params: CNNParams
    gen_learning_rate: float
    crtc_learning_rate: float
    crtc_weight_threshold: float
    n_critic: int


class WGANGPParams(_NamedTuple):
    gen_params: GeneratorParams
    crtc_params: CNNParams
    gen_learning_rate: float
    crtc_learning_rate: float
    gp_weight: float
    n_critic: int


def symmetric_params(ec_params: CoderParams):
    net_p = ec_params.net_params

    match net_p:
        case CNNParams():
            return ec_params._replace(net_params=_symmetric_params_cnn(net_p))
        case MLPParams():
            return ec_params._replace(net_params=_symmetric_params_mlp(net_p))
        case [CNNParams(), MLPParams()]:
            return ec_params._replace(net_params=_symmetric_params_cnn_mlp(net_p))
        case _:
            raise ValueError('invalid network parameters')


def _symmetric_params_mlp(net_p: BasicNNParams):
    return MLPParams(
        in_feature=net_p.out_features[-1],
        out_features=net_p.out_features[-2::-1] + [net_p.in_feature],
        bad_setting=net_p.bad_setting
    )


def _symmetric_params_cnn(net_p: BasicNNParams):
    return CNNParams(
        in_channel=net_p.out_channels[-1],
        out_channels=net_p.out_channels[-2::-1] + [net_p.in_channel],
        shape=_util.estimate_conv2d_size(net_p.shape, net_p.strides),
        kernels=net_p.kernels[::-1],
        strides=net_p.strides[::-1],
        bad_setting=net_p.bad_setting
    )


def _symmetric_params_cnn_mlp(net_p: BasicNNParams, mode='transposed'):
    if mode == 'transposed':
        cnn_p, mlp_p = net_p
        shape = _util.estimate_output_shape(cnn_p)
        mlp_p = mlp_p._replace(in_feature=_util.flatten_length(shape))
        return (_symmetric_params_mlp(mlp_p),
                _symmetric_params_cnn(cnn_p))


def _get_conv_type_from_params(params: BasicNNParams):
    conv_type = 'transposed'
    match params:
        case (CNNParams(up_sampling=u) |
              [_, CNNParams(up_sampling=u)]) if u is not None:
            conv_type = 'upsampling'
    return conv_type
