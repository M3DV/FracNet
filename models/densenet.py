from fastai.basics import *


__all__ = ['DenseNet']


class DenseNet(nn.Module):
    def __init__(self, in_channels, layers=(2, 4, 8, 6), growth_rate=16, bn_size=1):
        super().__init__()
        self.planes = in_channels

        self.layer1 = self._make_layer(layers[0], growth_rate, bn_size, 1)
        self.layer2 = self._make_layer(layers[1], growth_rate, bn_size, 2)
        self.layer3 = self._make_layer(layers[2], growth_rate, bn_size, 2)
        self.layer4 = self._make_layer(layers[3], growth_rate, bn_size, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, layers, growth_rate, bn_size, stride=2):
        block = []

        block.append(DenseBlock(layers, self.planes, growth_rate, bn_size))
        self.planes = self.planes + layers * growth_rate

        block.append(Transition(self.planes, self.planes // 2, stride))
        self.planes = self.planes // 2

        model = nn.Sequential(*block)
        model.out_channels = self.planes

        return model

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1, x2, x3, x4


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size):
        super().__init__(
            nn.Conv3d(in_channels, bn_size * growth_rate, 1, bias=False),
            nn.BatchNorm3d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv3d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False),
            nn.BatchNorm3d(growth_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        o = super().forward(x)
        o = torch.cat([x, o], dim=1)
        return o


class DenseBlock(nn.Sequential):
    def __init__(self, layers, in_channels, growth_rate, bn_size):
        super().__init__(*[
            DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size)
            for i in range(layers)
        ])


class Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
