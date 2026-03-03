import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeechEstimator(nn.Module):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha

        # Two Conv layers
        self.conv1 = nn.Conv2d(2, 32, kernel_size=(1,3))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1,3))

        # Pointwise conv layer to restore desired output shape
        self.mask_conv = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, real_mag_mask, noisy_phase, noisy_signal_real, noisy_signal_imag, magnitude_comp):
        #calculating the real and imaginary parts of the intermediate feature representation
        Y_real = real_mag_mask * torch.cos(noisy_phase)
        Y_imag = real_mag_mask * torch.sin(noisy_phase)

        # Concatenate along channel dimension
        x = torch.stack([Y_real, Y_imag], dim=1)

        # unsure how to implement stage about the less computationally complex CNN


        # Estimating Complex Mask
        mask = self.mask_conv(x)

        mask_real = mask[:, 0, :, :] # extracts the real part
        mask_imag = mask[:, 1, :, :] # extracts the imaginary part

        # CRM-based Multiplication
        '''
        S_real = mask_real * noisy_signal_real - mask_imag * noisy_signal_imag
        S_imag = mask_real * noisy_signal_imag + mask_imag * noisy_signal_real
        '''

        M_mag = torch.sqrt(mask_real**2 + mask_imag**2)
        M_phase = torch.atan2(mask_imag,mask_real)
        #limit magnitude mask to [0, 1]
        M_mag = torch.tanh(M_mag)

        theta = noisy_phase + M_phase

        complex_e = torch.cos(theta) + 1j * torch.sin(theta)

        # magnitude_comp = torch.sqrt(real_comp**2 + imag_comp**2)
        #outputed from the powerlaw compression in stage 1
        S_hat = magnitude_comp * M_mag * complex_e
        S_real = S_hat.real
        S_imag = S_hat.imag


        # Power-law Decompression
        decompressed_S_real = torch.sign(S_real) * torch.pow(torch.abs(S_real), 1/self.alpha)
        decompressed_S_imag = torch.sign(S_imag) * torch.pow(torch.abs(S_imag), 1/self.alpha)

        clean_speech_estimate = torch.complex(decompressed_S_real, decompressed_S_imag)

        return clean_speech_estimate


