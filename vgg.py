"""
vgg 11/13/16/19 in pytorch

please see the paper"Very Deep Convolutional Networks for Large-Scale Image Recognition" for more details.
"""
import torch
import torch.nn as nn

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.feature = self._make_layers(cfg[vgg_name])
        self.linear = nn.Sequential(
            nn.Linear(512, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 1000)
        )
    
    def forward(self, x):
        out = self.feature(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for i in cfg:
            if i == 'M':
                layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
            else:
                layers += [nn.Conv2d(in_channels, i, kernel_size=3, padding=1),
                            nn.BatchNorm2d(i),
                            nn.ReLU(inplace=True)]
                in_channels = i
        layers += [nn.AvgPool2d(kernel_size = 7, stride = 7)]
        return nn.Sequential(*layers)    

if __name__ == '__main__':
    net = VGG('vgg11')
    x = torch.randn(2, 3, 227, 227)
    out = net(x)
    print(out, out.shape)



