import functools as _funct

from torch import nn as _nn


def append_pbad_layers(cls=None, *, bn_layer=None):
    """ Append Pooling BatchNorm, Activation, Dropout layers after `self.layers` """
    if cls is None:
        return _funct.partial(append_pbad_layers, bn_layer=bn_layer)

    init = cls.__init__

    def __init__(
            self,
            *args,
            pooling=None,
            batchnorm=False,
            activation=None,
            dropout=0,
            **kwargs
    ):
        init(self, *args, **kwargs)

        # Assumes that the second element of args is out_channels.
        out_channels = args[1]

        if pooling is not None:
            self.pooling = pooling
            self.layers.append(self.pooling)

        if batchnorm:
            if bn_layer is None:
                raise ValueError('bn_layer should not be None when batchnorm=True')
            self.batch_norm = bn_layer(out_channels)
            self.layers.append(self.batch_norm)

        if activation is not None:
            self.activation = activation
            self.layers.append(self.activation)

        if dropout:
            self.dropout = _nn.Dropout(dropout)
            self.layers.append(self.dropout)

    cls.__init__ = __init__
    return cls


def sequential_forward(cls):
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    cls.forward = forward

    return cls
