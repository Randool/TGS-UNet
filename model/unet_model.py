import torch
import torch.nn as nn
import torch.nn.functional as F


class Double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Double_conv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.layers = nn.Sequential(nn.MaxPool2d(2), Double_conv(in_ch, out_ch))

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()
        self.conv = Double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode="bilinear", align_corners=True)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, ch=32):
        super(UNet, self).__init__()
        self.inLayers = nn.Sequential(
            nn.Conv2d(3, ch, 3), nn.Conv2d(ch, ch, 4)
        )

        self.down1 = Down(ch, 2 * ch)
        self.down2 = Down(2 * ch, 4 * ch)
        self.down3 = Down(4 * ch, 8 * ch)

        self.up1 = Up(12 * ch, 6 * ch)
        self.up2 = Up(8 * ch, 4 * ch)
        self.up3 = Up(5 * ch, 5 * ch // 2)

        self.outLayer = nn.Conv2d(5 * ch // 2, 1, 5)    # 需要拓展为 1@101*101

    def forward(self, x):
        x1 = self.inLayers(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = F.interpolate(x, scale_factor=1.1, mode="bilinear", align_corners=True)
        x = self.outLayer(x)
        x = torch.sigmoid(x)
        return x
