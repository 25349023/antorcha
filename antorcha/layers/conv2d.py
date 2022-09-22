from collections.abc import Sequence

from torch import nn as _nn

from .util import sequential_forward, append_bad_layers
from ..utils.aux_types import square_2d

__all__ = ['autopad_conv2d', 'autopad_conv_transpose2d', 'AutoConv2d', 'AutoConvTranspose2d']


def _pair(x):
    return (x, x) if not isinstance(x, Sequence) else x


def autopad_conv2d(
        in_channels: int,
        out_channels: int,
        kernel_size: square_2d[int],
        stride: square_2d[int]
) -> _nn.Conv2d:
    kernel_size = _pair(kernel_size)
    pad = tuple((k - 1) // 2 for k in kernel_size)

    return _nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=pad)


def autopad_conv_transpose2d(
        in_channels: int,
        out_channels: int,
        kernel_size: square_2d[int],
        stride: square_2d[int],
) -> _nn.ConvTranspose2d:
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)

    pad = tuple((k - 1) // 2 for k in kernel_size)
    out_padding = tuple(s + 2 * p - k for k, s, p in zip(kernel_size, stride, pad))

    return _nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                               padding=pad, output_padding=out_padding)


@sequential_forward
@append_bad_layers(bn_layer=_nn.BatchNorm2d)
class AutoConv2d(_nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: square_2d[int],
            stride: square_2d[int],
            padding=-1,
    ):
        super().__init__()

        self.layers = []
        if padding == -1:
            self.conv = autopad_conv2d(in_channels, out_channels, kernel_size, stride)
        else:
            self.conv = _nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding)
        self.layers.append(self.conv)


@sequential_forward
@append_bad_layers(bn_layer=_nn.BatchNorm2d)
class AutoConvTranspose2d(_nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: square_2d[int],
            stride: square_2d[int],
            padding: square_2d[int] = -1,
            out_padding=0
    ):
        super().__init__()

        self.layers = []
        if padding == -1:
            self.conv_t = autopad_conv_transpose2d(
                in_channels, out_channels, kernel_size, stride)
        else:
            self.conv_t = _nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, output_padding=out_padding)
        self.layers.append(self.conv_t)
