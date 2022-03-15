from functools import wraps as _wraps, partial as _partial

from torch import nn as _nn


def append_bad_layers(cls=None, *, bn_layer=None):
    """ Append BatchNorm, Activation, Dropout layers after `self.layers` """
    if cls is None:
        return _partial(append_bad_layers, bn_layer=bn_layer)

    init = cls.__init__

    @_wraps(init)
    def __init__(
            self,
            *args,
            batchnorm=False,
            activation=None,
            dropout=0,
            **kwargs
    ):
        init(self, *args, **kwargs)

        # Assumes that the second element of args is out_channels.
        out_channels = args[1]

        if batchnorm:
            if bn_layer is None:
                raise ValueError('bn_layer should not be None when batchnorm=True')
            self.batch_norm = bn_layer(out_channels)
            self.layers.append(self.batch_norm)

        if activation is not None and not isinstance(activation, _nn.Identity):
            self.activation = activation
            self.layers.append(self.activation)

        if dropout:
            self.dropout = _nn.Dropout(dropout)
            self.layers.append(self.dropout)

    cls.__init__ = __init__
    return cls


def sequential_forward(cls):
    """Adding `forward` method that simply forwards through `self.layers`"""

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    cls.forward = forward

    return cls
