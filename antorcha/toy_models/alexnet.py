import torch as _torch
from torch import nn as _nn

from .util import CNNParams as _CNNParams, MLPParams as _MLPParams, BADSettings as _Bad
from .basic_nn import CNN as _CNN, MLP as _MLP


class TinyAlexNet(_nn.Module):
    metric_names = ['Accuracy']

    def __init__(self, in_shape):
        super().__init__()
        cnn_param = _CNNParams(
            in_channel=3, out_channels=[96, 256, 384, 384, 256], shape=in_shape,
            kernels=[11, 5, 3, 3, 3], strides=[2, 2, 1, 1, 2],
            bad_setting=_Bad(batchnorm=True, activation=_nn.ReLU)
        )
        self.cnn = _CNN(cnn_param)

        self.flatten = _nn.Flatten()
        mlp_param = _MLPParams(
            in_feature=self.cnn.fmap_shape ** 2 * cnn_param.out_channels[-1],
            out_features=[4096, 4096, 10], bad_setting=_Bad(activation=_nn.ReLU)
        )
        self.mlp = _MLP(mlp_param)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x

    def loss(self, loss, pred, gt):
        return loss(pred, gt)

    def metrics(self, pred: _torch.Tensor, gt: _torch.Tensor):
        return (pred.argmax(dim=1) == gt).sum().item() / pred.size(0),
