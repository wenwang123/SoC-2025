"""Depthwise smoothing block shared by the mask decoders."""
from torch import nn


class _Smooth(nn.Sequential):
    def __init__(self, channels: int):
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
