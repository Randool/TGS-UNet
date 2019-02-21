import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from capsuleLayer import DigitCaps, PrimaryCaps


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return F.relu(self.conv(x))


class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()

        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()

    def forward(self, x):
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        return x
