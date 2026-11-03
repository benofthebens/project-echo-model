import soundfile as sf
import torch

TARGET_SR = 16000

def load_audio(path):
    """substitute for torchaudio.load; returns (waveform [channels, samples], sample_rate)."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr

def save_audio(path, waveform, sample_rate):
    """substitute for torchaudio.save."""
    sf.write(path, waveform.detach().cpu().numpy().T, sample_rate)