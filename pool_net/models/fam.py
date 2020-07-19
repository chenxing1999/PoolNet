from torch import nn
import torch
from torch.nn import functional as F


class FeatureAggregation(nn.Module):
    def __init__(self, in_channel, out_channel):
        """ Feature Aggregation Module
        Args:
            in_channel (int)
            out_channel (int)
        """
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
        """
        Args
            x  (tensor: B x C x W x H): Feature tensor
            x2 (tensor: B x C x W' x H'):
            
            with C = self.in_channel

        Return:
            if x2 is not None:
                res (tensor: B x C' x W' x H')         
            else:
                res (tensor: B x C' x W x H)
            with C' = self.out_channel
        """
        res = x
        for layer in self.layers:
            # Feed forward x to each layer
            y = layer(x)

            # Resize that output back to `x` then add to `res`
            ## Dont use += (inplace). Tensor will be need for backward
            res = res + F.interpolate(
                y, x.shape[2:], mode="bilinear", align_corners=True
            )

        res = F.relu(res)

        # Reshape res to x2 if x2 is not None
        if x2 is not None:
            res = F.interpolate(
                res, x2.shape[2:], mode="bilinear", align_corners=True
            )

        # Conv layer to mapping from in_channel to out_channel
        # (For merging information)
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
        """
        Args:
            x : (tensor B x C' x W' x H')
            x2: (tensor B x C  x W  x H )
            x3: (tensor B x C  x W  x H )
        Return:
            res: (tensor B x C x W x H)

        with C' = in_channel
             C  = out_channel
        """
        res = self.fam(x, x2)
        res = res + x2 + x3
        res = self.conv_sum_c(res)
        return res
