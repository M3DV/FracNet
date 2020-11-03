from fastai.basics import *


__all__ = ['ASPP']


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super().__init__()
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )])

        for rate in rates:
            self.convs.append(ASPPConv(in_channels, out_channels, rate))
        self.convs.append(ASPPPooling(in_channels, out_channels))

        self.trans = nn.Sequential(
            nn.Conv3d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.5)
        )

    def forward(self, x):
        o = [conv(x) for conv in self.convs]
        o = self.trans(torch.cat(o, 1))
        return o


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, rate):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-3:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='trilinear', align_corners=False)
