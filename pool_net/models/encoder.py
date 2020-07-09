from torch import nn
import torch
from torch.nn import functional as F

from torchvision import models

from .ppm import PyramidPooling


class VggBase(nn.Module):
    """ Warper for VGG backbone """

    def __init__(self, pretrain=False):
        super(VggBase, self).__init__()

        # Get Baseline of model
        base = models.vgg16(pretrain)
        self.base = base.features

        self.hook_index = [8, 15, 22, 29]

    def forward(self, x):
        """ 
        Args:
            x (tensor - B x C x W x H): Images
        Return:
            outs (list of tensor): List of encoded features
        """

        outs = []
        for idx, layer in enumerate(self.base):
            x = layer(x)
            if idx in self.hook_index:
                outs.append(x)
        return outs


class Encoder(nn.Module):
    """ Encoder class """

    def __init__(self, in_channel=512, output_channels=[512, 256, 128]):
        super(Encoder, self).__init__()

        self.base = VggBase()

        self.in_channel = in_channel
        self.out_channels = output_channels
        infos = []
        self.ppm = PyramidPooling(in_channel)

        for out_channel in self.out_channels:
            layer = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
            )
            infos.append(layer)
        self.infos = nn.ModuleList(infos)

        # Note here original author define init
        # weight. But I use default init of PyTorch

    def forward(self, x):
        """ Forward of Encoder 
        Args:
            x (tensor - B x C x W x H) - Input Image
        Return:
            xs (list tensor): List encoded feature by base
            
            infos (list tensor): Resized of P for concatenate
        """
        b, c, w, h = x.size()
        xs = self.base(x)

        p_tensor = self.ppm(xs[-1])
        xs_reverse = xs[::-1]

        infos = []
        for layer, feat in zip(self.infos, xs_reverse):
            t = F.interpolate(
                p_tensor, feat.size()[2:], mode="bilinear", align_corners=True
            )
            t = layer(t)
            infos.append(t)
        return xs, infos
