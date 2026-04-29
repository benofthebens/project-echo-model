import torch
import torch.nn as nn
import torch.nn.functional as Funct

from .. import features

from .conv_block.signal_processor import SignalProcessor  
from .clean_speech.speech_estimator import SpeechEstimator
from .subband.mask_estimator import MagnitudeMaskEstimator

class Denoiser(nn.Module):
    def __init__(self, alpha=0.3):
        """
        Implements the first stage of the audio processing pipeline: depthwise conv -> FGRU -> pointwise conv
        """
        # inputs: compressed magnitude spectrogram ~Xm
        # process: Conv -> FGRU + pointwise -> Subband GRU -> FC layers
        # outputL -Mm (shape: [frames, freq_bins)
        super().__init__()

        self.alpha = alpha
        
        # TODO: channelwise feature reorientation

        self.stage1_features = SignalProcessor()  
        self.mag_mask_estimator = MagnitudeMaskEstimator()
        self.speech_estimator = SpeechEstimator(alpha=alpha)

    def forward(self, x):
        # step 1 | input preprocessing 
        print(f'maximum value: {x.max()}, minimum value: {x.min()}')
        # extract waveform from spectrogram and apply power-law compression
        x_real, x_imag = features.stft_waveform(x)
        
        real_comp, imag_comp, magnitude_comp, phase_comp = features.power_law_compression(x_real, x_imag, alpha=self.alpha)

        # store original frequency resolution (257) and time frames (T)
        B, F_orig, T_orig = magnitude_comp.shape

        # TODO: implement reorientation of features
        # x = FeatureReorientation(real_comp, imag_comp, magnitude_comp, phase_comp)

        # input to conv is 4D [batch, channels, freq_bins, time]
        x_in = magnitude_comp.unsqueeze(1)  # add channel dimension for conv layers
        
        # shape remains [batch, channels, freq_bins, time]
        x_features = self.stage1_features(x_in)

        _, C, F_p, T_p = x_features.shape
        x_reshaped = x_features.reshape(B, C, -1)

        # B, C, F, T = x_features.shape
        # x_reshaped = x_features.reshape(B, C, F*T)  # reshape to [batch, channels, freq_bins * time]

        mask = self.mag_mask_estimator(x_reshaped)
        # mask = mask.reshape(B, 1, F, T)  # reshape back to [batch, channels, freq_bins, time], freq ~32 
        mask = mask.reshape(B, F_orig, F_p, T_p)

        # average the 'pooled frequency' dimension (F_p=32) - [B, 257, 1, T_p]
        mask = torch.mean(mask, dim=2, keepdim=True)

        # permute to [B, 1, 257, T_p], so it looks like a 1-channel spectrogram
        mask = mask.permute(0, 2, 1, 3)

        # interpolate time back to T_orig (e.g., 7 frames -> 63 frames)
        
        if mask.shape[3] != T_orig:
            # check if the time dimension (dim 3) needs stretching
            mask = Funct.interpolate(
                mask, size=(F_orig, T_orig), mode='bilinear', align_corners=False
            )

        # match the frequency dimension of magnitude_comp
        # if mask.shape[2] != magnitude_comp.shape[1]:
        #   mask = Funct.interpolate(mask, size=(magnitude_comp.shape[1], T), mode='bilinear', align_corners=False)

        concat_Y = features.intermediate_feature_construction(mask, phase_comp)
      
        clean_complex = self.speech_estimator(concat_Y)

        # apply power law compressed features to speech estimator -> real_mag_mask, noisy_phase, noisy_signal_real, noisy_signal_imag, magnitude_comp
        S_hat = features.clean_speech_processing(clean_complex, magnitude_comp, phase_comp)
        
        # apply power law decompression to the output of the speech estimator
        S_final = features.power_law_decompression(S_hat, alpha=self.alpha)

        S_final_inverse = torch.istft(S_final, n_fft=512, hop_length=256, window=torch.hann_window(512).to(x.device), length=x.shape[-1])

        # test whether increasing magnitude helps make audio audible
        return S_final_inverse * 132