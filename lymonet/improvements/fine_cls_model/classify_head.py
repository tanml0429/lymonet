
import torch
import torch.nn as nn


from lymonet.apis.yolov8_api import Conv



class EchoHead(nn.Module):
    """对于输入的特征图，输出是否有高回声区域的分类结果"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """
        c2: 2, 2 classification for with or without high echo area
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class Classify(nn.Module):
    """LymoNet classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLOv8 classification head with specified input and output channels, kernel size, stride,
        padding, and groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

        self.echo_head = EchoHead(c1, 2, k, s, p, g)


    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        # x = self.conv(x)  # (bs, 1280, 20, 20)
        # x = self.pool(x)  # (bs, 1280, 1, 1)
        # x = x.flatten(1)  # (bs, 1280)
        # x = self.drop(x)  # (bs, 1280)
        # x = self.linear(x)  # (bs, 3)
        cls_x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        cls_x = cls_x if self.training else cls_x.softmax(1)
        if self.training:
            echo_x = self.echo_head(x)
            return [cls_x, echo_x]
        else:
            return cls_x
        
    




