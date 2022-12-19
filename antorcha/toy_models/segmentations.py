import itertools as _iter

import torch as _torch
from torch import nn as _nn
from torchvision import models as _models

from .. import layers as _layers
from ..train import metric as _metric


class ResnetExtract(_nn.Module):
    def __init__(self):
        super().__init__()

        resnet = _torch.hub.load('pytorch/vision:v0.10.0', 'resnet18',
                                 weights=_models.ResNet18_Weights.IMAGENET1K_V1)
        self.blocks = []
        for name, module in list(resnet.named_children())[:-2]:
            setattr(self, name, module)
            self.blocks.append(module)

    def forward(self, img):
        outputs = tuple(_iter.accumulate(
            self.blocks, lambda x, layer: layer(x), initial=img
        ))

        outputs = outputs[2:3] + outputs[-4:]
        return outputs


class UpConvLayer(_nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride, up_sample):
        super().__init__()
        self.upsampling = _nn.Upsample(scale_factor=up_sample, mode='nearest')
        self.conv = _layers.AutoConv2d(in_ch, out_ch, k_size, stride, batchnorm=True, activation=_nn.ReLU())

    def forward(self, x):
        out = self.upsampling(x)
        out = self.conv(out)
        return out


class TinyUNet(_nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.resnet = ResnetExtract()
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.upconvs = _nn.Sequential(
            UpConvLayer(512, 256, 3, 1, 2),
            UpConvLayer(512, 128, 3, 1, 2),
            UpConvLayer(256, 64, 3, 1, 2),
            UpConvLayer(128, 64, 3, 1, 2),
            UpConvLayer(128, 32, 3, 1, 2),
            _layers.AutoConv2d(32, num_class, 1, 1),
        )
        self.num_class = num_class

        self.metric = _metric.MeanIoU(self.num_class)

    def forward(self, img):
        resnet_out = self.resnet(img)
        ups = self.upconvs

        out = ups[0](resnet_out[-1])
        out = ups[1](_torch.cat([resnet_out[-2], out], dim=1))
        out = ups[2](_torch.cat([resnet_out[-3], out], dim=1))
        out = ups[3](_torch.cat([resnet_out[-4], out], dim=1))
        out = ups[4](_torch.cat([resnet_out[-5], out], dim=1))
        out = ups[5](out)

        return out

    def loss(self, loss, pred, gt):
        return loss(pred, gt)


class PPM(_nn.Module):
    def __init__(self, input_channels, adaptive_sizes=None):
        super().__init__()
        adaptive_sizes = adaptive_sizes or (1, 2, 3, 6)
        self.layers = []
        for a_size in adaptive_sizes:
            self.layers.append(_nn.Sequential(
                _nn.AdaptiveAvgPool2d(a_size),
                _nn.Conv2d(input_channels, input_channels // 4, kernel_size=1),
                _nn.Upsample(size=7, mode='bilinear', align_corners=True)
            ))
        self.layers = _nn.ModuleList(self.layers)

    def forward(self, x):
        feat = [x]
        for layer in self.layers:
            feat.append(layer(x))
        out = _torch.cat(feat, dim=1)
        return out


class TinyPSPNet(_nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.resnet = ResnetExtract()
        self.ppm = PPM(512)
        self.upconvs = _nn.Sequential(
            UpConvLayer(1024, 256, 3, 1, 2),
            UpConvLayer(256, 128, 3, 1, 2),
            UpConvLayer(128, 64, 3, 1, 2),
            UpConvLayer(64, 64, 3, 1, 2),
            UpConvLayer(64, 32, 3, 1, 2),
            _layers.AutoConv2d(32, num_class, 1, 1),
        )

        self.num_class = num_class
        self.metric = _metric.MeanIoU(self.num_class)

    def forward(self, img):
        resnet_out = self.resnet(img)
        ups = self.upconvs

        out = ups[0](self.ppm(resnet_out[-1]))
        out = ups[1](resnet_out[-2] + out)
        out = ups[2](resnet_out[-3] + out)
        out = ups[3](resnet_out[-4] + out)
        out = ups[4](resnet_out[-5] + out)
        out = ups[5](out)

        return out

    def loss(self, loss, pred, gt):
        return loss(pred, gt)
