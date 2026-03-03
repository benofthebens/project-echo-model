import torch

# package containing all feature processing methods to be used in the pipeline


def stft_waveform(waveform):
    stft = torch.stft(waveform, n_fft=512, hop_length=256, win_length=512,return_complex=True,
                      window=torch.hann_window(window_length=512,device=waveform.device))
    
    # Extract real & imaginary parts
    x_real = stft.real
    x_imag = stft.imag
    return x_real, x_imag

def power_law_compression(x_real, x_imag,alpha):
    real_comp = torch.sign(x_real) * torch.abs(x_real)**alpha
    imag_comp = torch.sign(x_imag) * torch.abs(x_imag)**alpha

    # Reconstruct compressed magnitude and phase, input features
    magnitude_comp = torch.sqrt(real_comp**2 + imag_comp**2)
    phase_comp = torch.atan2(imag_comp, real_comp)

    return real_comp, imag_comp, magnitude_comp, phase_comp

def power_law_decompression(S_complex,alpha):
    real = torch.sign(S_complex.real) * torch.abs(S_complex.real)**(1/alpha)
    imag = torch.sign(S_complex.imag) * torch.abs(S_complex.imag)**(1/alpha)

    return torch.complex(real,imag)

def intermediate_feature_construction(magnitude_mask, phase_comp):
    real = magnitude_mask * torch.cos(phase_comp)
    imag = magnitude_mask * torch.sin(phase_comp)

    return torch.cat([real, imag], dim=1)

def clean_speech_processing(M, x_mag, x_phase):
    '''
    s_real = M.real * x_real
    s_imag = M.imag * x_imag
    return torch.cat([s_real, s_imag], dim=1)
    '''
    # or DCCRN-E

    M_mag = torch.sqrt(M.real**2 + M.imag**2)
    M_phase = torch.atan2(M.imag,M.real)

    #limit magnitude mask to [0, 1]
    M_mag = torch.tanh(M_mag)

    theta = x_phase + M_phase

    complex_e = torch.cos(theta) + 1j * torch.sin(theta)

    S_hat = x_mag * M_mag * complex_e
    return S_hat
