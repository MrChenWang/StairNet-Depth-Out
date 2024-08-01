import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce


class Passthrough(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, width=1.0):
        super(Passthrough, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(int(c1 * 4 * width), int(c2 * width), k, 1),
            nn.BatchNorm2d(int(c2 * width)),
            nn.ReLU(),
        )

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, width=1.0):
        super(Focus, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1 * 4, int(c2 * width), k, 1),
            nn.BatchNorm2d(int(c2 * width)),
            nn.ReLU(),
        )

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class ASPP(nn.Module):
    def __init__(self, in_channels, width=1.0):
        super(ASPP, self).__init__()
        internal_channel = int(in_channels * width // 2)
        self.branch1 = nn.Sequential(
            nn.Conv2d(int(in_channels * width), internal_channel, kernel_size=3, stride=1, groups=16, padding=(2, 6),
                      dilation=(2, 6)),
            nn.BatchNorm2d(internal_channel),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(int(in_channels * width), internal_channel, kernel_size=3, stride=1, groups=16, padding=(3, 12),
                      dilation=(3, 12)),
            nn.BatchNorm2d(internal_channel),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(int(in_channels * width), internal_channel, kernel_size=3, stride=1, groups=16, padding=(5, 18),
                      dilation=(5, 18)),
            nn.BatchNorm2d(internal_channel),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(int(in_channels * width), internal_channel, kernel_size=3, stride=1, groups=16, padding=(6, 24),
                      dilation=(6, 24)),
            nn.BatchNorm2d(internal_channel),
            nn.ReLU(inplace=True)
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(6 * internal_channel, int(in_channels * width), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(in_channels * width)),
            nn.ReLU(inplace=True)
        )
        self.SE_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(int(in_channels * width), int(in_channels * 0.25 * width), 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels * 0.25 * width), int(in_channels * width), 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        f4 = self.branch4(x)
        out = torch.cat((f1, f2, f3, f4, x), dim=1)
        out = self.conv1_1(out)
        f_se = self.SE_branch(out)
        out = out * f_se
        return out


class ResBlockX_SE(nn.Module):
    def __init__(self, inchannel, outchannel, stride=(1, 1), se_ratio=0.25, num_groups=32, shrink_ratio=0.5, padding=1,
                 dilation=1, width=1.0):
        super(ResBlockX_SE, self).__init__()
        internal_channel = int(outchannel * shrink_ratio * width)
        # if num_groups > internal_channel:
        #    num_groups = internal_channel
        self.left = nn.Sequential(
            nn.Conv2d(int(inchannel * width), internal_channel, 1, 1, 0),
            nn.BatchNorm2d(internal_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_channel, internal_channel, kernel_size=3, stride=stride, padding=padding,
                      dilation=dilation, groups=int(num_groups * width),
                      bias=False),
            nn.BatchNorm2d(internal_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_channel, int(outchannel * width), 1, 1, 0),
            nn.BatchNorm2d(int(outchannel * width))
        )
        self.SE_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(int(outchannel * width), int(outchannel * se_ratio * width), 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outchannel * se_ratio * width), int(outchannel * width), 1, 1, 0),
            nn.Sigmoid()
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(int(inchannel * width), int(outchannel * width), kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(int(outchannel * width))
            )

    def forward(self, x):
        out = self.left(x)
        f_se = self.SE_branch(out)
        out = out * f_se
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResBlockX_SE_Super1_1(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, se_ratio=0.25, num_groups=16, shrink_ratio=0.25, padding=2,
                 dilation=2, width=1.0):
        super(ResBlockX_SE_Super1_1, self).__init__()
        internal_channel = int(outchannel * shrink_ratio * width)
        self.left1_2 = nn.Sequential(
            nn.Conv2d(int(inchannel * width), internal_channel, 1, 1, 0),
            nn.BatchNorm2d(internal_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_channel, internal_channel, kernel_size=3, stride=stride, padding=(padding, padding),
                      dilation=(dilation, dilation), groups=int(num_groups * width),
                      bias=False),
            nn.BatchNorm2d(internal_channel),
            nn.ReLU(inplace=True),
        )
        self.left2_1 = nn.Sequential(
            nn.Conv2d(int(inchannel * width), internal_channel, 1, 1, 0),
            nn.BatchNorm2d(internal_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_channel, internal_channel, kernel_size=3, stride=stride,
                      padding=(int(padding * 0.2 + 1), padding), dilation=(int(dilation * 0.2 + 1), dilation),
                      groups=int(num_groups * width),
                      bias=False),
            nn.BatchNorm2d(internal_channel),
            nn.ReLU(inplace=True),
        )
        self.channel_expand = nn.Sequential(
            nn.Conv2d(int(2 * internal_channel), int(outchannel * width), 1, 1, 0),
            nn.BatchNorm2d(int(outchannel * width))
        )
        self.SE_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(int(outchannel * width), int(outchannel * se_ratio * width), 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outchannel * se_ratio * width), int(outchannel * width), 1, 1, 0),
            nn.Sigmoid()
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(int(inchannel * width), int(outchannel * width), kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(int(outchannel * width))
            )

    def forward(self, x):
        out = torch.cat((self.left2_1(x), self.left1_2(x)), dim=1)
        out = self.channel_expand(out)
        f_se = self.SE_branch(out)
        out = out * f_se
        out += self.shortcut(x)  # 注意这里是直接＋的而不是cat，直接加需要特征图大小和通道数都相同
        out = F.relu(out)
        return out


class SEmoudle(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=1, L=32, width=1.0):
        super(SEmoudle, self).__init__()
        d = max(int(in_channels * width) // r, L)
        self.M = M
        self.out_channels = int(out_channels * width)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv2d(self.out_channels, d, 1, bias=False),
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Conv2d(d, self.out_channels * M, 1, 1, bias=False)  # 升维
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        output = []
        output.append(x1)
        output.append(x2)
        # the part of fusion
        U = reduce(lambda x, y: x + y, output)
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        a_b = self.softmax(a_b)
        # the part of selection
        a_b = list(a_b.chunk(self.M, dim=1))  # split to a and b
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1), a_b))
        V = list(map(lambda x, y: x * y, output, a_b))
        V = reduce(lambda x, y: x + y, V)
        return V
