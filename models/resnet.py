from fastai.basics import *


__all__ = ['ResNet', 'BasicBlock', 'Bottleneck']


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, 1, stride=stride, bias=False)


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.norm1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes)
        self.norm2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(inplanes, planes)
        self.norm1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.norm2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.norm3 = nn.BatchNorm3d(planes * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out, inplace=True)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes=None, block=BasicBlock, layers=(2, 2, 2, 2), n=64):
        super().__init__()
        self.planes = in_channels

        self.layer1 = self._make_layer(block, 1 * n, layers[0], 1)
        self.layer2 = self._make_layer(block, 2 * n, layers[1], 2)
        self.layer3 = self._make_layer(block, 4 * n, layers[2], 2)
        self.layer4 = self._make_layer(block, 8 * n, layers[3], 2)

        if num_classes is not None:
            self.fc = nn.Linear(block.expansion * 8 * n, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.planes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.planes, planes, stride, downsample))
        self.planes = planes * block.expansion
        layers.extend([block(self.planes, planes)] * (blocks - 1))

        module = nn.Sequential(*layers)
        module.out_channels = self.planes

        return module

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if hasattr(self, 'fc'):
            x = F.adaptive_avg_pool3d(x4, 1)
            x = self.fc(x.flatten(1)).flatten()
            return x

        return x1, x2, x3, x4
