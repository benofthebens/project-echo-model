import os
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from src.modules.denoiser import Denoiser

from audio_utils import load_audio, TARGET_SR

# set to True for quick test run with a small subset of the dataset
TEST_RUN = True 
MAX_FILES = 4 if TEST_RUN else 100
EPOCHS = 2 if TEST_RUN else 10
MODEL_PATH = "models/denoiser_test.pth" if TEST_RUN else "models/denoiser.pth"


class NoiseCancellationDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, target_sample_rate=TARGET_SR):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        # ensure files are sorted to maintain alignment between noisy and clean pairs
        self.files = sorted(os.listdir(noisy_dir))[:MAX_FILES]
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        noisy_path = os.path.join(self.noisy_dir, filename)
        clean_path = os.path.join(self.clean_dir, filename)

        noisy, noisy_sr = load_audio(noisy_path)
        clean, clean_sr = load_audio(clean_path)

        noisy = self.preprocess(noisy, noisy_sr)
        clean = self.preprocess(clean, clean_sr)

        min_len = min(noisy.shape[1], clean.shape[1])
        noisy = noisy[:, :min_len]
        clean = clean[:, :min_len]

        return noisy, clean

    def preprocess(self, waveform, sample_rate):
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != self.target_sample_rate:
            resampler = T.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)

        return waveform


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NoiseCancellationDataset(
        noisy_dir="data/noisy_trainset_wav", clean_dir="data/clean_trainset_wav"
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = Denoiser(alpha=0.3).to(device)
    print("Trainable params:", sum(p.numel() for p in model.parameters()))

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for noisy, clean in loader:

            # squeeze to remove channel dimension and move to device
            noisy = noisy.squeeze(1).to(device)
            clean = clean.squeeze(1).to(device)

            output = model(noisy)
            loss = loss_fn(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # track total loss for the epoch
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.6f}")
    os.makedirs("models", exist_ok=True)

    torch.save(model.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    main()
