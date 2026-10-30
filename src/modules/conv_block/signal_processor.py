import torch
import torch.nn as nn
from .fgru import FGRU
from .depthwise_conv import DepthwiseConv

class SignalProcessor(nn.Module):
    def __init__(self, num_freqs=257,subband_size=48):
        """
        Implements the first stage of the audio processing pipeline: depthwise conv -> FGRU -> pointwise conv
        """
        # inputs: compressed magnitude spectrogram ~Xm
        # process: Conv -> FGRU + pointwise -> Subband GRU -> FC layers
        # outputL -Mm (shape: [frames, freq_bins)
        super().__init__()
        
        # TODO: channelwise feature reorientation - use magnitude comp from power law compression (may implement in new module)

        self.depthwise1 = DepthwiseConv(in_channels=1, out_channels=32)
        self.depthwise2 = DepthwiseConv(in_channels=32, out_channels=64)
        self.depthwise3 = DepthwiseConv(in_channels=64, out_channels=96)
        self.depthwise4 = DepthwiseConv(in_channels=96, out_channels=128)
        self.FGRU = FGRU(in_channels=128, hidden_size=64)
        self.pointwise = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)

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