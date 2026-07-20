import torch
from torch import nn

from src.model.VAE.modules.layers import ResnetBlock3D


class AttributeNeck(nn.Module):
    def __init__(self, channels=192, num_blocks=1):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                ResnetBlock3D(in_channels=channels, out_channels=channels)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, features: torch.Tensor):
        return self.blocks(features)
