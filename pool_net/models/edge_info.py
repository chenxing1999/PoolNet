from torch import nn
import torch
from torch.nn import functional as F


class EdgeInfo(nn.Module):
    def __init__(self, c_in, c_out):
        """
        Args:
            c_in (int): In channel
            c_out (int): Out Channel
        """
        super(EdgeInfo, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.layers = nn.Sequential(
                nn.Conv2d(c_in, c_in, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
            )
        
    def forward(self, xs, feat_size, mode="bilinear"):
        """
        Args:
            xs (List of Tensor)
            feat_size (Tensor size): (W, H) of original feature

            mode (str): Support bilinear, nearest

        Return:
            feat (Tensor - B x c_out x W x H)
        """

        feat = [
            F.interpolate(x, feat_size, mode=mode, align_corners=True)
            for x in xs
        ]

        feat = torch.cat(feat, dim=1)
        return self.layers(feat)

class EdgeResidualBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(EdgeResidualBlock, self).__init__()
        self.c_in = c_in
        self.c_out = c_out

        quarter_c_in = c_in // 4
        self.block1 = nn.Sequential(
            # Bottle neck block to compress information
            nn.Conv2d(c_in, quarter_c_in, 1, bias=False),
            nn.Conv2d(quarter_c_in, quarter_c_in, 3, 1, 1, bias=False),
            nn.Conv2d(quarter_c_in, c_in, 1, bias=False),
        )

        self.block2 = nn.Sequential(
            # Bottle neck block to compress information
            nn.Conv2d(c_in, quarter_c_in, 1, bias=False),
            nn.Conv2d(quarter_c_in, quarter_c_in, 3, 1, 1, bias=False),
            nn.Conv2d(quarter_c_in, c_in, 1, bias=False),
        )

        self.out_conv = nn.Conv2d(c_in, c_out, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x (tensor - B x c_in x W x H)
        Return:
            x (tensor - B x c_out x W x H)
        """
        x = x + self.block1(x)
        x = self.relu(x)
        
        x = x + self.block2(x)
        x = self.relu(x)

        return self.out_conv(x)

