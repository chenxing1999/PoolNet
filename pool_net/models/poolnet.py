from torch import nn
import torch
from torch.nn import functional as F

from .encoder import Encoder
from .fam import FeatureAggregation, DeepPoolLayer
from .edge_info import EdgeInfo, EdgeResidualBlock


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


class VggPoolNetEdge(nn.Module):
    def __init__(self):
        super(VggPoolNetEdge, self).__init__()
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
        self.score = nn.Conv2d(256, 1, 1)

        # Edge feature layers

        ## Convert feat to edge feat
        self.res_blocks = nn.ModuleList(
            [
                EdgeResidualBlock(512, 16),
                EdgeResidualBlock(256, 16),
                EdgeResidualBlock(128, 16),
            ]
        )

        ## Find edge from edge feat
        self.score_edges = nn.ModuleList(
            [
                nn.Conv2d(16, 1, 1), 
                nn.Conv2d(16, 1, 1), 
                nn.Conv2d(16, 1, 1),
            ]
        )

        self.final_edge_score = nn.Conv2d(3, 1, 1)

        self.edge_info = EdgeInfo(48, 128)

    def forward(self, x, infer_edge=False):
        x_size = x.size()

        # encoded_feat: list of Encoded feature by VGG
        # infos: list of P-tensor resized with different size
        encoded_feat, infos = self.encoder(x)

        # Reverse encoded feat list
        encoded_feat = encoded_feat[::-1]

        edge_merge = []
        prev = self.fams[0](encoded_feat[0], encoded_feat[1], infos[0])
        edge_merge.append(prev)

        for k in range(1, len(encoded_feat) - 1):
            prev = self.fams[k](prev, encoded_feat[k + 1], infos[k])
            edge_merge.append(prev)

        edge_feat = [
            block(feat) for block, feat in zip(self.res_blocks, edge_merge)
        ]

        if infer_edge:
            edge_out = [score(feat) for score, feat in zip(self.score_edges, edge_feat)]
            edge_out = [F.interpolate(edge, x_size[2:], mode="bilinear", align_corners=True) for edge in edge_out]
            edge_mask = self.final_edge_score(torch.cat(edge_out, dim=1))

        merge = self.last_fam(prev)

        # Concate all edge feature to single feat map
        edge_feat = self.edge_info(edge_feat, merge.size()[2:])

        merge = torch.cat([merge, edge_feat], dim=1)
        merge = self.score(merge)

        merge = F.interpolate(
            merge, x_size[2:], mode="bilinear", align_corners=True
        )


        if infer_edge:
            return merge, edge_mask, edge_out
        else:
            return merge
