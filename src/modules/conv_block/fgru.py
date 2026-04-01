import torch.nn as nn

class FGRU(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(FGRU, self).__init__()

        # Reduce channel dimension before GRU
        self.pre_linear = nn.Linear(in_channels, hidden_size)

        # GRU for temporal modelling
        self.gru = nn.GRU(
            input_size=hidden_size, # unsure what the input size should be
            hidden_size=hidden_size,
            num_layers=1, #use of a single layer reduces computational complexity
            batch_first=True, #reshapes input data to [B, T, hidden_size * 2] since its bidirectional
            bidirectional=True #the FGRU is described as bidirectional in the paper
        )

        # 3. Maps FGRU back to the original feature dimension expected
        self.post_linear = nn.Linear(hidden_size * 2, in_channels)

        self.activation = nn.ReLU()
        #this line isn't explicitly mentioned in the paper but necessary so that
        #there is non linearity

    def forward(self, x):
        """
        x: (batch, channels, freq, time)
        """

        B, C, F, T = x.shape
        out = x.permute(0, 2, 3, 1).reshape(B * F, T, C)
        # 1. Compress features
        out = self.pre_linear(out)

        # 2. Run GRU over time dimension
        out, _ = self.gru(out)

        # 3. Expand GRU output back to original channel size
        out = self.post_linear(out)
        out = out.reshape(B, F, T, C).permute(0, 3, 1, 2)  # reshape back to (batch, channels, freq, time)

        # 4. Fuse GRU features with original features
        out = self.activation(out + x)

        return out

# if __name__ == "__main__":
#   fgru = FGRU(128, 64)

#   tensor = torch.rand(10, 5, 128)
#   x = fgru(tensor)
#   print(x.shape)