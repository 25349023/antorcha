from torch import nn as _nn

from .util import sequential_forward, append_pbad_layers


@sequential_forward
@append_pbad_layers(bn_layer=_nn.BatchNorm1d)
class AutoLinear(_nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = _nn.Linear(in_features, out_features)
        self.layers = [self.linear]

