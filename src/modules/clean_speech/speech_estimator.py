import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeechEstimator(nn.Module):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha

        # Two Conv layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=(1,3), padding=(0,1)) # TODO - input channels = 4 works on test_Pipeline but input channels = 2 is needed for real audio
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1,3), padding=(0,1))

        # Pointwise conv layer to restore desired output shape
        self.mask_conv = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x):  

        # TODO: implement stage about the less computationally complex CNN

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # output is [B, 2, F, T]
        mask = self.mask_conv(x)

        # extract real and imaginary parts of the mask
        mask_real = mask[:, 0, :, :]
        mask_imag = mask[:, 1, :, :]

        
        return torch.complex(mask_real, mask_imag)


