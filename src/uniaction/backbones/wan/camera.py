# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for UniAction; upstream copyright and Apache-2.0 attribution are in NOTICE.
from __future__ import annotations

import torch.nn as nn


class SimpleAdapter(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride,
        num_residual_blocks=1,
        downscale_factor=8,
    ):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=downscale_factor)
        self.conv = nn.Conv2d(
            in_dim * downscale_factor**2,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_dim) for _ in range(num_residual_blocks)]
        )

    def forward(self, x):
        batch, channels, frames, height, width = x.size()
        del channels
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(batch * frames, -1, height, width)
        x = self.pixel_unshuffle(x)
        x = self.conv(x)
        x = self.residual_blocks(x)
        x = x.view(batch, frames, x.size(1), x.size(2), x.size(3))
        return x.permute(0, 2, 1, 3, 4)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x + residual
