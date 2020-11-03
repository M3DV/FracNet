from fastai.basics import *


__all__ = ['UNetPP']


class UNetPP(nn.Module):
    def __init__(self, in_channels, num_classes, n=16):
        super().__init__()

        self.pool = nn.MaxPool3d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.conv0_0 = ConvBlock(in_channels, n)
        self.conv1_0 = ConvBlock(n, 2 * n)
        self.conv2_0 = ConvBlock(2 * n, 4 * n)
        self.conv3_0 = ConvBlock(4 * n, 8 * n)

        self.conv0_1 = ConvBlock(3 * n, n)
        self.conv1_1 = ConvBlock(6 * n, 2 * n)
        self.conv2_1 = ConvBlock(12* n, 4 * n)

        self.conv0_2 = ConvBlock(4 * n, n)
        self.conv1_2 = ConvBlock(8 * n, 2 * n)

        self.conv0_3 = ConvBlock(5 * n, n)
        self.segment = nn.Conv3d(n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x = self.segment(x0_3)

        return x


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
