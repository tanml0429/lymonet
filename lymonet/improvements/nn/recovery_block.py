

import torch
from torch import nn

from .nn import CBAM

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