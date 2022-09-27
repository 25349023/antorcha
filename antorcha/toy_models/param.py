from typing import NamedTuple as _NamedTuple, Callable as _Callable, Union as _Union

from torch import nn as _nn

from .util import estimate_conv2d_size


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


class CoderParams(_NamedTuple):
    net_params: _Union[MLPParams, CNNParams]
    z_dim: int


class GeneratorParams(_NamedTuple):
    net_params: _Union[MLPParams, CNNParams]
    z_dim: int


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

    if isinstance(net_p, CNNParams):
        return CoderParams(
            net_params=CNNParams(
                in_channel=net_p.out_channels[-1],
                out_channels=net_p.out_channels[-2::-1] + [net_p.in_channel],
                shape=estimate_conv2d_size(net_p.shape, net_p.strides),
                kernels=net_p.kernels[::-1],
                strides=net_p.strides[::-1],
                bad_setting=net_p.bad_setting
            ),
            z_dim=ec_params.z_dim,
        )
    elif isinstance(net_p, MLPParams):
        return CoderParams(
            net_params=MLPParams(
                in_feature=net_p.out_features[-1],
                out_features=net_p.out_features[-2::-1] + [net_p.in_feature],
                bad_setting=net_p.bad_setting
            ),
            z_dim=ec_params.z_dim
        )
