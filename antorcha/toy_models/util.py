import math as _math


def estimate_conv2d_size(in_size: int, strides: list[int],
                         up_sample: list[int] = None):
    up_sample = up_sample or [1] * len(strides)
    for s, u in zip(strides, up_sample):
        in_size *= u
        in_size = _math.ceil(in_size / s)
    return in_size


def attach_device_prop(cls):
    def getter(self):
        return self._device

    def setter(self, value):
        self._device = value
        self.to(value)

    cls.device = property(getter, setter)
    return cls
