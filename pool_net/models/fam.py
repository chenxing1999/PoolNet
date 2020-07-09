from torch import nn
import torch
from torch.nn import functional as F


class FeatureAggregation(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeatureAggregation, self).__init__()
        pool_size = [2, 4, 8]

        layers = []
        for kernel_size in pool_size:
            layer = nn.Sequential(
                nn.AvgPool2d(kernel_size, stride=kernel_size),
                nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=False),
            )
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.conv_sum = nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False)

    def forward(self, x, x2=None):
        res = x
        for layer in self.layers:
            y = layer(x)
            res = torch.add(
                y,
                F.interpolate(
                    y, x.shape[2:], mode="bilinear", align_corners=True
                ),
            )
        res = F.relu(res)

        if x2:
            res = F.interpolate(
                res, x2.shape[2:], mode="bilinear", align_corners=True
            )
        res = self.conv_sum(res)
        return res


class DeepPoolLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DeepPoolLayer, self).__init__()
        self.fam = FeatureAggregation(in_channel, out_channel)
        self.conv_sum_c = nn.Conv2d(
            out_channel, out_channel, 3, 1, 1, bias=False
        )

    def forward(self, x, x2, x3):
        res = self.fam(x, x2)
        res = res + x2 + x3
        res = self.conv_sum_c(res)
        return res
