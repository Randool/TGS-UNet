import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

cuda = False


def squash(inputs):
    norm_2 = (inputs ** 2).sum(-1, keepdim=True)
    outputs = norm_2 * inputs / ((1 + norm_2) * torch.sqrt(norm_2))
    return outputs


class PrimaryCaps(nn.Module):
    def __init__(self, num_caps, in_channels, out_channels, kernel_size):
        super(PrimaryCaps, self).__init__()
        
        self.capsules = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=0)
                for _ in range(num_caps)
            ]
        )

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), 32 * 6 * 6, -1)  # 【】
        return squash(u)


class DigitCaps(nn.Module):
    def __init__(self, num_caps, num_routes, in_channels, out_channels):
        super(DigitCaps, self).__init__()

        self.num_caps = num_caps
        self.num_routes = num_routes
        self.in_channels = in_channels

        self.W = nn.Parameter(
            torch.randn(1, num_routes, num_caps, out_channels, in_channels)
        )

    def forward(self, x):
        # 获得中间向量
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_caps, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_caps, 1))
        if cuda:
            b_ij = b_ij.cuda()
        
        # 动态路由
        num_iters = 3
        for iteration in range(num_iters):
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = squash(s_j)

            if iteration < num_iters - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)
