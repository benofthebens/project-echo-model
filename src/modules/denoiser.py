import torch
import torch.nn as nn

import src.features as features

from clean_speech.speech_estimator import SpeechEstimator

from conv_block.signal_processor import SignalProcessor
from conv_block.feature_reorientation import FeatureReorientation

from src.modules.conv_block.depthwise_conv import DepthwiseConv
from subband.mask_estimator import MagnitudeMaskEstimator


from conv_block.fgru import FGRU

class Denoiser(nn.Module):
    def __init__(self):
        """
        Implements the first stage of the audio processing pipeline: depthwise conv -> FGRU -> pointwise conv
        """
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

    def forward(self, x):
        # step 1 | input preprocessing 

        # extract waveform from spectrogram and apply power-law compression
        x_real, x_imag = features.stft_waveform(x)
        
        real_comp, imag_comp, magnitude_comp, phase_comp = features.power_law_compression(x_real, x_imag, alpha=0.3)

        # TODO: implement reorientation of features
        x = FeatureReorientation(real_comp, imag_comp, magnitude_comp, phase_comp)
        x = SignalProcessor()
        mask = MagnitudeMaskEstimator()

        # calculate the intermediate feature computations and concatenate along the channel dimension
        concat_Y = features.intermediate_feature_construction(mask, phase_comp)

        # apply power law compressed features to speech estimator -> real_mag_mask, noisy_phase, noisy_signal_real, noisy_signal_imag, magnitude_comp
        x = SpeechEstimator(concat_Y, phase_comp, x_real, x_imag, magnitude_comp)

        x = features.clean_speech_processing(x_real, x_imag, real_comp, imag_comp)
        
        # apply power law decompression to the output of the speech estimator
        x = features.power_law_decompression(x, alpha=0.3)


       # step 2 | convulution block

        return x