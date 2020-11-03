from fastai.basics import *


__all__ = ['ResUNet']


class ResUNet(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.decoder = Decoder(encoder)
        self.segment = nn.Conv3d(encoder.layer1.out_channels, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        size = x.shape[-3:]
        x = self.segment(self.decoder(*self.encoder(x)))
        x = F.interpolate(x, size=size, mode='trilinear', align_corners=False)
        return x


class Decoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.up1 = UpConv(encoder.layer4.out_channels, encoder.layer3.out_channels)
        self.up2 = UpConv(encoder.layer3.out_channels, encoder.layer2.out_channels)
        self.up3 = UpConv(encoder.layer2.out_channels, encoder.layer1.out_channels)

    def forward(self, x1, x2, x3, x4):
        x = self.up1(x3, x4)
        x = self.up2(x2, x)
        x = self.up3(x1, x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x = self.conv2(x2)
        x = self.conv1(torch.cat([x1, x], dim=1))
        return x
