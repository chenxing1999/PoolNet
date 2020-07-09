from torch import nn
import torch
from torch.nn import functional as F


class VggPoolNet(nn.Module):
    def __init__(self):
        super(VggPoolNet, self).__init__()
        self.encoder = None
        self.fams = None
        self.score = None

    def forward(self, x):
        x_size = x.size()
        encoded_feat, infos = self.encoder(x)

        encoded_feat = encoded_feat[::-1]

        prev = self.fams[0](encoded_feat[0], encoded_feat[1], infos[0])

        for k in range(1, len(conv2merge) - 1):
            prev = self.fams[k](prev, encoded_feat[k + 1], infos[k])

        merge = self.fams[-1](prev)
        merge = self.score(merge, x_size)
        return merge
