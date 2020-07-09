from pool_net.models.encoder import Encoder
import torch


def test_encoder():
    encoder = Encoder()
    x = torch.rand(size=(1, 3, 224, 224))
    encoder(x)
