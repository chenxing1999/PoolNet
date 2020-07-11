from torch import nn
import torch
from torch.nn import functional as F


class PyramidPooling(nn.Module):
    """ Pyradmid Pooling Module """

    def __init__(self, in_channel=512):
        super(PyramidPooling, self).__init__()
        self.in_channel = in_channel
        ppms = []
        for kernel_size in [1, 3, 5]:
            layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(kernel_size),
                nn.Conv2d(in_channel, in_channel, 1, 1, bias=False),
                nn.ReLU(inplace=True),
            )
            ppms.append(layer)

        self.ppms = nn.ModuleList(ppms)

        # Concate output of ppms by ppms_cat
        self.ppms_cat = nn.Sequential(
            nn.Conv2d(in_channel * 4, in_channel, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, interpolate_mode="bilinear"):
        """ Forward function of PPM
        Args:
            x (tensor - Shape: B x C x W x H)
            with C = self.in_channel
            
            interpolate_mode (str): Support nearest and bilinear
            Note: In original paper of PPM they use nearest :lol:
                But PoolNet use bilinear. So I set bilinear as default

        Return:
            out (tensor - Shape: B x C x W x H)
        """

        b, c, w, h = x.size()

        list_out = [x]
        for layer in self.ppms:
            o = layer(x)

            # Resize feature tensor to original size.
            o = F.interpolate(
                o, size=(w, h), mode=interpolate_mode, align_corners=True
            )
            list_out.append(o)

        # Concatenate output
        list_out = torch.cat(list_out, dim=1)
        output = self.ppms_cat(list_out)
        return output
