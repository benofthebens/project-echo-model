import torch
import torch.nn as nn
from FGRU import FGRU

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


class CRN(nn.Module):
    def __init__(self, num_freqs=257,subband_size=48):
        # inputs: compressed magnitude spectrogram ~Xm
        # process: Conv -> FGRU + pointwise -> Subband GRU -> FC layers
        # outputL -Mm (shape: [frames, freq_bins)
        super().__init__()
        
        # TODO: channelwise feature reorientation

        self.depthwise1 = DepthwiseConv(in_channels=1, out_channels=32)
        self.depthwise2 = DepthwiseConv(in_channels=32, out_channels=64)
        self.depthwise3 = DepthwiseConv(in_channels=64, out_channels=96)
        self.depthwise4 = DepthwiseConv(in_channels=96, out_channels=128)
        self.FGRU = FGRU(in_channels=128, hidden_size=64, bidirectional=True, batch_first=True)
        self.pointwise = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)

        # TODO: incorporate subband gru and FC layers


    def forward(self, x):

        x = self.depthwise1(x)
        x = self.depthwise2(x)
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = self.depthwise3(x)
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = self.depthwise4(x)
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = self.FGRU(x)
        x = self.pointwise(x)

        return x