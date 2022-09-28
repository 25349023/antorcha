import functools as _funct
import math as _math
import operator as _opr

from . import param as _param


def estimate_conv2d_size(in_size: int, strides: list[int] = None,
                         up_sample: list[int] = None):
    if strides is None and up_sample is None:
        raise ValueError('must provide at least one of strides or up_sample')

    strides = strides or [1] * len(up_sample)
    up_sample = up_sample or [1] * len(strides)
    for s, u in zip(strides, up_sample):
        in_size *= u
        in_size = _math.ceil(in_size / s)
    return in_size


def flatten_length(shape: tuple) -> int:
    return _funct.reduce(_opr.mul, shape)


def estimate_output_shape(cnn_params: _param.CNNParams):
    shape = ([estimate_conv2d_size(cnn_params.shape, cnn_params.strides)] * 2
             + [cnn_params.out_channels[-1]])
    return shape


def attach_device_prop(cls):
    def getter(self):
        return self._device

    def setter(self, value):
        self._device = value
        self.to(value)

    cls.device = property(getter, setter)
    return cls
