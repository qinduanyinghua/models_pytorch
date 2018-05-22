"""
MobileNetV2 in pytorch

please see the paper "MobileNetV2: Inverted Residuals and Linear Bottlenecks" for more details
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 

class Block(nn.Module):
    # pointwise + depthwise + pointwise
    def __init__(self, expansion, in_channels, out_channels, stride):
        super(Block, self).__init__()
        self.stride = stride

        inter_channels = expansion * in_channels
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1,padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3,stride=stride,padding=1,groups=inter_channels,bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv3 = nn.Conv2d(inter_channels, out_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,kernel_size=1,stride=stride,padding=0,bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = F.relu6(self.bn3(self.conv3(out)))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out

class MobileNetV2(nn.Module):
    # (expansion, out_channels, repeat times, stride)
    cfg = [(1, 16, 1, 1),
           (6, 24, 2, 2),
           (6, 32, 3, 2),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.layers = self._make_layers(in_channels=32)
        self.conv2 = nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=1280)
        self.linear = nn.Linear(in_features=1280, out_features=num_classes)
    
    def _make_layers(self, in_channels = 32):
        layers = []
        for expansion, out_channels, repeat_time, stride in self.cfg:
            strides = [stride] + [1] * (repeat_time - 1)
            for stride in strides:
                layers.append(Block(expansion,in_channels,out_channels,stride))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.layers(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out

if __name__ == "__main__":
    net = MobileNetV2()
    print(net)
    x = torch.randn((2,3,224,224))
    y = net(x)
    print(y.size())
