from fastai.basics import *


__all__ = ['SELayer']


class SELayer(nn.Sequential):
    def __init__(self, channels, reduction=16):
        super().__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = super(SELayer, self).forward(x)
        x = w * x
        return x
