import torch
import torchaudio
import torchaudio.transforms as T
from src.modules.denoiser import Denoiser

def run_inference(input_path, output_path):
    # load denoiser model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Denoiser(alpha=0.3).to(device)
    model.eval()

    # load audio file - waveform shape: [channels, samples]
    waveform, sample_rate = torchaudio.load(input_path)
    waveform = waveform.to(device)

    print(f"> loaded {input_path} | sample rate: {sample_rate} | shape: {waveform.shape}")

    # preprocessing (mModel expects 16kHz mono [batch, samples]), coonverting to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # resample to 16kHz if necessary
    if sample_rate != 16000:
        resampler = T.Resample(sample_rate, 16000).to(device)
        waveform = resampler(waveform)
        print(f"> resampled to 16000Hz")


    with torch.no_grad():
        print("> processing audio...")
        # model expects [batch, samples but torchaudio gives [1, samples], so we can just pass it directly as a batch of 1
        denoised_waveform = model(waveform)

    # save output and ensure it's on CPU for saving
    denoised_waveform = denoised_waveform.cpu()
    torchaudio.save(output_path, denoised_waveform, 16000)
    print(f"> successful - denoised audio saved to: {output_path}")

if __name__ == "__main__":
    # absolute paths to input and output files - TODO change to relative paths if necessary
    INPUT_FILE = "data/noisy/noisy.wav"
    OUTPUT_FILE = "data/denoised/denoised.wav"
    
    try:
        run_inference(INPUT_FILE, OUTPUT_FILE)
    except Exception as e:
        print(f"> inference failed: {e}")