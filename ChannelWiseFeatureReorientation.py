import torch
import torch.nn as nn

class ChannelwiseFeatureReorientation(nn.Module):
    def __init__(self, freq_num = 257, subband_num = 48, overlap = 1/3):
        super().__init__()

        self.freq_num = freq_num
        self.subband_num = subband_num
        self.interval = int(subband_num * (1 - overlap))

    def forward(self, x):
        #[B, C, T, F] -> [B, C, T x F]
        subbands = []

        # sliding window implementation
        for start in range(0, self.freq_num - self.subband_num + 1, self.interval):
            end = start + self.subband_num
            y = x[:, :, start:end] # selects only the frequencies between start and end

            subbands.append(y)

        # concatenate along channel dimension
        x = torch.cat(subbands, dim = 1)

        return x



def split_subbands(x, num_bands):
    """
    x: (B, F, T)
    returns: (B, K, F', T)
    """
    B, F, T = x.shape
    assert F % num_bands == 0, f"Frequency dimension F ({F}) must be divisible by num_bands ({num_bands})"

    band_size = F // num_bands
    bands = []

    for i in range(num_bands):
        bands.append(x[:, i*band_size:(i+1)*band_size, :])

    return torch.stack(bands, dim=1)  # channel dim

def merge_subbands(x):
    """
    x: (B, K, F', T)
    returns: (B, F, T)
    """
    B, K, Fp, T = x.shape
    return x.reshape(B, K * Fp, T)


# Corrected input tensor where the second dimension (F) is divisible by num_bands
# For example, using a random tensor of shape (1, 4, 10)
input = torch.randn(1, 1, 257)

cwr = ChannelwiseFeatureReorientation()

x = cwr(input)
print(x)
