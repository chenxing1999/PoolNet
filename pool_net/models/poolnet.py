from torch import nn
import torch
from torch.nn import functional as F

from .encoder import Encoder
from .fam import FeatureAggregation, DeepPoolLayer


class VggPoolNet(nn.Module):
    def __init__(self):
        super(VggPoolNet, self).__init__()
        in_channel = 512
        out_channels = [512, 256, 128]
        self.encoder = Encoder(in_channel, out_channels)

        self.fams = nn.ModuleList(
            [
                DeepPoolLayer(512, 512),
                DeepPoolLayer(512, 256),
                DeepPoolLayer(256, 128),
            ]
        )

        self.last_fam = FeatureAggregation(128, 128)
        self.score = nn.Conv2d(128, 1, 1, stride=1)

    def forward(self, x):
        x_size = x.size()

        # encoded_feat: list of Encoded feature by VGG
        # infos: list of P-tensor resized with different size
        encoded_feat, infos = self.encoder(x)

        # Reverse encoded feat list
        encoded_feat = encoded_feat[::-1]

        prev = self.fams[0](encoded_feat[0], encoded_feat[1], infos[0])

        for k in range(1, len(encoded_feat) - 1):
            prev = self.fams[k](prev, encoded_feat[k + 1], infos[k])

        merge = self.last_fam(prev)
        merge = self.score(merge)

        merge = F.interpolate(
            merge, x_size[2:], mode="bilinear", align_corners=True
        )
        return merge
