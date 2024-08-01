from nets.modules import *


class StairNet_DepthOut(nn.Module):
    def __init__(self, width=1.0):
        super(StairNet_DepthOut, self).__init__()
        self.depth_backbone = Depth_backbone(width=width)
        self.backbone = Backbone(width=width)

    def forward(self, x):
        x1, x4, f = self.depth_backbone(x)
        fb1, fb2, fr1, fr2, fm = self.backbone(x1, f)
        return fb1, fb2, fr1, fr2, fm, x4


class Depth_backbone(nn.Module):
    def __init__(self, width=1.0):
        super(Depth_backbone, self).__init__()
        self.initial = Focus(3, 64, width=width)
        self.resblock1 = nn.Sequential(
            ResBlockX_SE(64, 128, stride=2, width=width),
            ResBlockX_SE(128, 128, width=width),
            ResBlockX_SE(128, 128, width=width)
        )
        self.resblock2 = nn.Sequential(
            ResBlockX_SE(128, 256, stride=2, width=width),
            ResBlockX_SE_Super1_1(256, 256, padding=2, dilation=2, width=width),
            ResBlockX_SE(256, 256, width=width),
            ResBlockX_SE_Super1_1(256, 256, padding=4, dilation=4, width=width),
            ResBlockX_SE(256, 256, width=width),
            ResBlockX_SE_Super1_1(256, 256, padding=8, dilation=8, width=width),
            ResBlockX_SE(256, 256, width=width),
            ResBlockX_SE_Super1_1(256, 256, padding=16, dilation=16, width=width)
        )
        self.resblock3 = nn.Sequential(
            ResBlockX_SE(256, 512, stride=2, width=width),
            ResBlockX_SE_Super1_1(512, 512, padding=2, dilation=2, width=width),
            ResBlockX_SE(512, 512, width=width),
            ResBlockX_SE_Super1_1(512, 512, padding=4, dilation=4, width=width),
            ResBlockX_SE(512, 512, width=width),
            ResBlockX_SE_Super1_1(512, 512, padding=8, dilation=8, width=width),
            ResBlockX_SE(512, 512, width=width),
            ResBlockX_SE_Super1_1(512, 512, padding=16, dilation=16, width=width)
        )
        # neck输入为32 x 32 x 512
        self.depth_conv1 = nn.Sequential(
            nn.Conv2d(int(512 * width), int(256 * width), 3, 1, 1),
            nn.BatchNorm2d(int(256 * width)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(int(256 * width), int(256 * width), 4, 2, 1),
            nn.BatchNorm2d(int(256 * width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(256 * width), int(128 * width), 3, 1, 1),
            nn.BatchNorm2d(int(128 * width)),
            nn.ReLU(inplace=True)
        )  # 64x64x128
        self.depth_conv2 = nn.Sequential(
            nn.ConvTranspose2d(int(128 * width), int(128 * width), 4, 2, 1),
            nn.BatchNorm2d(int(128 * width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(128 * width), int(64 * width), 3, 1, 1),
            nn.BatchNorm2d(int(64 * width)),
            nn.ReLU(inplace=True)
        )  # 128x128x64
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(int(64 * width), int(128 * width), 1, 1, 0),
            nn.BatchNorm2d(int(128 * width))
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(int(256 * width), int(512 * width), 3, 2, 1),
            nn.BatchNorm2d(int(512 * width)),
            nn.Conv2d(int(512 * width), int(512 * width), 3, 1, 1),
            nn.BatchNorm2d(int(512 * width)),
            nn.Conv2d(int(512 * width), int(512 * width), 3, 2, 1),
            nn.BatchNorm2d(int(512 * width))
        )
        self.depth_conv3 = nn.Sequential(
            nn.ConvTranspose2d(int(64 * width), int(64 * width), 4, 2, 1),
            nn.BatchNorm2d(int(64 * width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(64 * width), int(32 * width), 3, 1, 1),
            nn.BatchNorm2d(int(32 * width)),
            nn.ReLU(inplace=True)
        )  # 256x256x32
        self.depth_conv4 = nn.Sequential(
            nn.ConvTranspose2d(int(32 * width), int(32 * width), 4, 2, 1),
            nn.BatchNorm2d(int(32 * width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(32 * width), 1, 1, 1, 0),
            nn.Sigmoid()
        )  # 512x512x1
        self.focus = Passthrough(32, 128, width=width)

    def forward(self, x):
        x1 = self.initial(x)
        x2 = self.resblock1(x1)
        x2 = self.resblock2(x2)
        x2 = self.resblock3(x2)
        x2 = self.depth_conv1(x2)
        x3 = self.depth_conv2(x2)
        x4 = self.depth_conv3(x3)
        f1 = self.focus(x4)
        x4 = self.depth_conv4(x4)

        f = torch.cat((self.conv1_1(x3), f1), dim=1)

        return x1, x4, self.conv1_2(f)


class Backbone(nn.Module):
    def __init__(self, width=1.0):
        super(Backbone, self).__init__()
        self.res1 = nn.Sequential(
            ResBlockX_SE(64, 256, stride=2, width=width),
            ResBlockX_SE(256, 256, width=width),
            ResBlockX_SE(256, 256, width=width)
        )
        self.res2 = nn.Sequential(
            ResBlockX_SE(256, 512, stride=2, width=width),
            ResBlockX_SE_Super1_1(512, 512, padding=2, dilation=2, width=width),
            ResBlockX_SE(512, 512, width=width),
            ResBlockX_SE_Super1_1(512, 512, padding=4, dilation=4, width=width),
            ResBlockX_SE(512, 512, width=width),
            ResBlockX_SE_Super1_1(512, 512, padding=8, dilation=8, width=width),
            ResBlockX_SE(512, 512, width=width),
            ResBlockX_SE_Super1_1(512, 512, padding=16, dilation=16, width=width)
        )
        self.res3 = nn.Sequential(
            ResBlockX_SE(512, 512, stride=2, width=width),
            ResBlockX_SE_Super1_1(512, 512, padding=2, dilation=2, width=width),
            ResBlockX_SE(512, 512, width=width),
            ResBlockX_SE_Super1_1(512, 512, padding=4, dilation=4, width=width),
            ResBlockX_SE(512, 512, width=width),
            ResBlockX_SE_Super1_1(512, 512, padding=8, dilation=8, width=width),
            ResBlockX_SE(512, 512, width=width),
            ResBlockX_SE_Super1_1(512, 512, padding=16, dilation=16, width=width)
        )
        self.selective = SEmoudle(512, 512, width=width)
        self.neck = nn.Sequential(
            nn.Conv2d(int(512 * width), int(256 * width), 3, stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(int(256 * width)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(int(256 * width), int(256 * width), kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(int(256 * width)),
            nn.ReLU(inplace=True)
        )
        self.neck_b = nn.Sequential(
            nn.Conv2d(int(256 * width), int(128 * width), 3, stride=1, padding=1),
            nn.BatchNorm2d(int(128 * width)),
            nn.ReLU(inplace=True)
        )
        self.neck_r = nn.Sequential(
            nn.Conv2d(int(256 * width), int(128 * width), 3, stride=1, padding=1),
            nn.BatchNorm2d(int(128 * width)),
            nn.ReLU(inplace=True)
        )
        self.conf_branch_b = nn.Sequential(
            nn.Conv2d(int(128 * width), int(128 * width), 3, 1, 1),
            nn.BatchNorm2d(int(128 * width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(128 * width), 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.loc_branch_b = nn.Sequential(
            nn.Conv2d(int(128 * width), int(128 * width), 3, 1, 1),
            nn.BatchNorm2d(int(128 * width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(128 * width), 4, 1, 1, 0),
            nn.Sigmoid()
        )
        self.conf_branch_r = nn.Sequential(
            nn.Conv2d(int(128 * width), int(128 * width), 3, 1, 1),
            nn.BatchNorm2d(int(128 * width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(128 * width), 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.loc_branch_r = nn.Sequential(
            nn.Conv2d(int(128 * width), int(128 * width), 3, 1, 1),
            nn.BatchNorm2d(int(128 * width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(128 * width), 4, 1, 1, 0),
            nn.Sigmoid()
        )
        self.mask_conv1 = nn.Sequential(
            nn.Conv2d(int(512 * width), int(256 * width), 3, 1, 1),
            nn.BatchNorm2d(int(256 * width)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(int(256 * width), int(256 * width), 4, 2, 1),
            nn.BatchNorm2d(int(256 * width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(256 * width), int(128 * width), 3, 1, 1),
            nn.BatchNorm2d(int(128 * width)),
            nn.ReLU(inplace=True)
        )  # 64x64x128
        self.mask_conv2 = nn.Sequential(
            nn.ConvTranspose2d(int(128 * width), int(128 * width), 4, 2, 1),
            nn.BatchNorm2d(int(128 * width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(128 * width), int(64 * width), 3, 1, 1),
            nn.BatchNorm2d(int(64 * width)),
            nn.ReLU(inplace=True)
        )  # 128x128x64
        self.mask_conv3 = nn.Sequential(
            nn.ConvTranspose2d(int(64 * width), int(64 * width), 4, 2, 1),
            nn.BatchNorm2d(int(64 * width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(64 * width), int(32 * width), 3, 1, 1),
            nn.BatchNorm2d(int(32 * width)),
            nn.ReLU(inplace=True)
        )  # 256x256x32
        self.mask_conv4 = nn.Sequential(
            nn.ConvTranspose2d(int(32 * width), int(32 * width), 4, 2, 1),
            nn.BatchNorm2d(int(32 * width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(32 * width), 3, 1, 1, 0)
        )  # 512x512x3

    def forward(self, x, f):
        x = self.res1(x)
        # 替换为直接cat试一下, 注意res2的输入通道数改为了512
        #x = torch.cat((x, f), dim=1)
        x = self.res2(x)
        x = self.res3(x)
        x = self.selective(x, f)
        fm = self.mask_conv1(x)
        fm = self.mask_conv2(fm)
        fm = self.mask_conv3(fm)
        fm = self.mask_conv4(fm)

        x = self.neck(x)
        f_b = self.neck_b(x)
        f_r = self.neck_r(x)
        fb1 = self.conf_branch_b(f_b)
        fb2 = self.loc_branch_b(f_b)
        fr1 = self.conf_branch_r(f_r)
        fr2 = self.loc_branch_r(f_r)
        return fb1, fb2, fr1, fr2, fm


if __name__ == "__main__":
    x1 = torch.randn(1, 3, 512, 512)
    stairnet = StairNet_DepthOut(width=1.0)
    stairnet.eval()
    # stat(resnet, (3, 512, 512))
    y = stairnet(x1)
    y1, y2, y3, y4, y5, y6 = y
    print(y1.size(), y2.size(), y3.size(), y4.size(), y5.size(), y6.size())
