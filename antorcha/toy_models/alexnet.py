import torch as _torch
from torch import nn as _nn

from .basic_nn import CNN as _CNN, MLP as _MLP
from .param import BADSettings as _Bad, MLPParams as _MLPParams, CNNParams as _CNNParams
from ..layers import autopad_conv2d as _autopad_conv2d


class TinyAlexNet(_nn.Module):
    """AlexNet without pooling, dropout, but adding batchnorm for performance"""

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


class AlexNet(_nn.Module):
    metric_names = ['accuracy']

    def __init__(self):
        super().__init__()
        self.model = _nn.Sequential(
            _autopad_conv2d(3, 96, 11, 1),
            _nn.ReLU(),
            _nn.MaxPool2d(3, 2),
            _autopad_conv2d(96, 256, 5, 1),
            _nn.ReLU(),
            _nn.MaxPool2d(3, 2),
            _autopad_conv2d(256, 384, 3, 1),
            _nn.ReLU(),
            _autopad_conv2d(384, 384, 3, 1),
            _nn.ReLU(),
            _autopad_conv2d(384, 256, 3, 1),
            _nn.ReLU(),
            _nn.MaxPool2d(3, 2),
            _nn.Flatten(),
            _nn.Linear(2304, 4096),
            _nn.ReLU(),
            _nn.Dropout(0.5),
            _nn.Linear(4096, 4096),
            _nn.ReLU(),
            _nn.Dropout(0.5),
            _nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def loss(self, loss, pred, gt):
        return loss(pred, gt)

    def metrics(self, pred: _torch.Tensor, gt: _torch.Tensor):
        return (pred.argmax(dim=1) == gt).sum().item() / pred.size(0),
