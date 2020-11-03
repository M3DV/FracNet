from fastai.basics import *
from .aspp import ASPP


__all__ = ['DeepLabV3p']


class DeepLabV3p(nn.Module):
    def __init__(self, backbone, num_classes, rates=(1, 2, 3)):
        super().__init__()
        self.backbone = backbone
        self.aspp = ASPP(self.backbone.layer4.out_channels, 128, rates)
        self.decoder = Decoder(self.backbone.layer2.out_channels, 128, 128, 64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        size = x.shape[-3:]
        o = self.backbone(x)
        x = self.decoder(o[1], self.aspp(o[3]))
        x = F.interpolate(x, size=size, mode='trilinear', align_corners=False)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels_low, out_channels_low, in_channels_high, out_channels_high, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels_low, out_channels_low, 1, bias=False),
            nn.BatchNorm3d(out_channels_low),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels_low + in_channels_high, out_channels_high, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels_high),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.5),
            nn.Conv3d(out_channels_high, out_channels_high, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels_high),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(out_channels_high, num_classes, 1, bias=True)
        )

    def forward(self, x1, x2):
        size = x1.shape[-3:]
        y = self.conv1(x1)
        x = F.interpolate(x2, size=size, mode='trilinear', align_corners=False)
        x = self.conv2(torch.cat((x, y), dim=1))
        return x
