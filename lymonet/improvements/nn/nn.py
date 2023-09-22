
import torch
import torch.nn as nn


# 自己加的融合模块
class RecoveryBlock(nn.Module):
    """Recovery Block"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, 1, 1)
        self.conv3 = nn.Conv2d(3, 3, 3, 1, 1)
        self.prelu = nn.PReLU()
        self.conv_in = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1),
                                     nn.PReLU())
        self.conv_down1 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1),
                                        nn.PReLU())
        self.conv_down2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1),
                                        nn.PReLU())
        self.conv_up2 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1),
                                      nn.PReLU(),
                                      nn.PixelShuffle(2))
        self.conv_up1 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1),
                                      nn.PReLU(),
                                      nn.PixelShuffle(2))
        self.conv_out = nn.Sequential(nn.Conv2d(64, 12, 3, 1, 1),
                                      nn.PReLU(),
                                      nn.PixelShuffle(2))
        # self.attention2 = SEModule(64)
        # self.attention1 = SEModule(32)
        self.attention2 = CBAM(64)
        self.attention1 = CBAM(32)

    def forward(self, x):
        _, c, _, _ = x.shape
        if c == 1:
            x = self.prelu(self.conv1(x))
        elif c == 3:
            x = self.prelu(self.conv3(x))
        x1 = self.conv_in(x)
        x2 = self.conv_down1(x1)
        x3 = self.conv_down2(x2)
        x4 = self.conv_up2(x3)
        x5 = self.conv_up1(torch.cat((x4, self.attention2(x2)), dim=1))
        x6 = self.conv_out(torch.cat((x5, self.attention1(x1)), dim=1))
        return x6


# 自己加的CAM
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


# 自己加的SAM
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


# 自己加的CBAM
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


# 自己加的ResNet with CBAM
class ResBlock_CBAM(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(ResBlock_CBAM, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )
        self.cbam = CBAM(channel=places * self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
