"""Inference architecture for the NAFNet GoPro width-32 checkpoint."""

import torch
from torch import nn
from torch.nn import functional as F


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, image):
        image = image.permute(0, 2, 3, 1)
        return F.layer_norm(image, self.weight.shape, self.weight, self.bias, 1e-6).permute(0, 3, 1, 2)


def gate(features):
    first, second = features.chunk(2, dim=1)
    return first * second


class NAFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        expanded = 2 * channels
        self.norm1 = ChannelNorm(channels)
        self.norm2 = ChannelNorm(channels)
        self.conv1 = nn.Conv2d(channels, expanded, 1)
        self.conv2 = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.conv4 = nn.Conv2d(channels, expanded, 1)
        self.conv5 = nn.Conv2d(channels, channels, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, image):
        features = gate(self.conv2(self.conv1(self.norm1(image))))
        residual = image + self.beta * self.conv3(features * self.sca(features))
        return residual + self.gamma * self.conv5(gate(self.conv4(self.norm2(residual))))


class NAFNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = (32, 64, 128, 256)
        self.intro = nn.Conv2d(3, widths[0], 3, padding=1)
        self.ending = nn.Conv2d(widths[0], 3, 3, padding=1)
        self.encoders = nn.ModuleList(
            nn.Sequential(*(NAFBlock(width) for _ in range(count)))
            for width, count in zip(widths, (1, 1, 1, 28))
        )
        self.downs = nn.ModuleList(nn.Conv2d(width, 2 * width, 2, stride=2) for width in widths)
        self.middle_blks = nn.Sequential(NAFBlock(512))
        self.ups = nn.ModuleList(
            nn.Sequential(nn.Conv2d(2 * width, 4 * width, 1, bias=False), nn.PixelShuffle(2))
            for width in reversed(widths)
        )
        self.decoders = nn.ModuleList(nn.Sequential(NAFBlock(width)) for width in reversed(widths))

    def forward(self, image):
        height, width = image.shape[-2:]
        padded = F.pad(image, (0, -width % 16, 0, -height % 16))
        features = self.intro(padded)
        skips = []
        for encode, down in zip(self.encoders, self.downs):
            features = encode(features)
            skips.append(features)
            features = down(features)
        features = self.middle_blks(features)
        for decode, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            features = decode(up(features) + skip)
        return (self.ending(features) + padded)[..., :height, :width]
