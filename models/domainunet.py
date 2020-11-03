from fastai.basics import *
from torch.autograd import Function


__all__ = ['DomainUNet']


class DomainUNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, n=16, alpha=0.5):
        super().__init__()
        FC = partial(FCL, num_classes=num_classes, alpha=alpha)
        self.first = ConvBlock(in_channels, n)
        self.fc1   = FC(n)
        self.down1 = Down(n, 2 * n)
        self.fc2   = FC(2 * n)
        self.down2 = Down(2 * n, 4 * n)
        self.fc3   = FC(4 * n)
        self.down3 = Down(4 * n, 8 * n)
        self.fc4   = FC(8 * n)
        self.up1   = Up(8 * n, 4 * n)
        self.fc5   = FC(4 * n)
        self.up2   = Up(4 * n, 2 * n)
        self.fc6   = FC(2 * n)
        self.up3   = Up(2 * n, n)
        self.fc7   = FC(n)
        self.final = nn.Conv3d(n, out_channels, 1)

    def forward(self, x):
        x1 = self.first(x)
        y1 = self.fc1(x1)
        x2 = self.down1(x1)
        y2 = self.fc2(x2)
        x3 = self.down2(x2)
        y3 = self.fc3(x3)
        x4 = self.down3(x3)
        y4 = self.fc4(x4)
        x  = self.up1(x4, x3)
        y5 = self.fc5(x)
        x  = self.up2(x, x2)
        y6 = self.fc6(x)
        x  = self.up3(x, x1)
        y7 = self.fc7(x)
        x  = self.final(x)
        return x, y1, y2, y3, y4, y5, y6, y7


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.MaxPool3d(2),
            ConvBlock(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, y):
        x = self.conv2(x)
        x = self.conv1(torch.cat([y, x], dim=1))
        return x


class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        alpha, = ctx.saved_tensors
        return alpha * grad_output.neg(), None


class FCL(nn.Sequential):
    def __init__(self, in_channels, num_classes, alpha, expansion=4):
        super().__init__(
            nn.Conv3d(in_channels, in_channels * expansion, 1, bias=False),
            nn.BatchNorm3d(in_channels * expansion),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1),
            nn.Linear(in_channels * expansion, num_classes)
        )
        self.alpha = alpha

    def forward(self, x):
        x = GRL.apply(x, self.alpha)
        x = super().forward(x)
        return x
