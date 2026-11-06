"""Conditional convolutional discriminator with spectral normalization."""

import torch
from torch import nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    def __init__(self, num_attrs=3):
        super().__init__()

        # Maps the attributes to a 64x64 map that is stacked on the image as a 4th channel.
        # Spectral normalization is applied to this layer as well.
        self.attr_conv = nn.Sequential(
            spectral_norm(nn.Linear(num_attrs, 64 * 64)),
            nn.ReLU(),
        )

        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(4, 64, 4, 2, 1)),      # 64x64 -> 32x32 (RGB + attribute map)
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),    # 32x32 -> 16x16
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),   # 16x16 -> 8x8
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),   # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            spectral_norm(nn.Linear(512 * 4 * 4, 1)),
            nn.Sigmoid(),                                  # probability that the image is real
        )

    def forward(self, img, attrs):
        # attrs: [B, 3] -> [B, 1, 64, 64]
        attr_map = self.attr_conv(attrs).view(-1, 1, 64, 64)
        x = torch.cat([img, attr_map], dim=1)
        return self.model(x)
