"""
mobilenet in pytorch

please see the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"for more details.
"""
import torch
import torch.nn as nn 
import torch.nn.functional as F 

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                kernel_size=3,stride=stride,padding=1,groups=in_channels,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                stride=1,padding=0,bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class MobileNet(nn.Module):
    cfg = [(64,1), (128,2), (128,1), (256,2), (256,1), (512,2), 
           (512,1), (512,1), (512,1), (512,1), (512,1), (1024,2), (1024,1)]
    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2,
                                padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)
        self.linear = nn.Linear(in_features=1024, out_features=self.num_classes)
    def _make_layers(self, in_channels=32):
        layers = []
        for out_channels, stride in self.cfg:
            layers.append(Block(in_channels=in_channels, out_channels=out_channels, stride=stride))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out,7)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out

if __name__ == "__main__":
    net = MobileNet()
    x = torch.randn((2,3,224,224))
    y = net(x)
    print(net)
    print(y.size())
