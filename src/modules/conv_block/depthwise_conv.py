import torch.nn as nn

class DepthwiseConv(nn.Module):
    #contains 4 depthwise separable conv layers
    #depthwise -> batch + ReLU -> pointwise -> batch + ReLU
    def __init__(self, in_channels, out_channels,stride=1):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3,stride=stride, padding=1,groups=in_channels),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True),
                                   )
    def forward(self, x):
        return self.block(x)