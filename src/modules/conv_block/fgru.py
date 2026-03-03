import torch
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
        x: (batch, time, channels)
        """
        # 1. Compress features
        x_proj = self.pre_linear(x)

        # 2. Run GRU over time dimension
        gru_out, _ = self.gru(x_proj)

        # 3. Expand GRU output back to original channel size
        out = self.post_linear(gru_out)

        # 4. Fuse GRU features with original features
        out = self.activation(out + x)

        return out

# if __name__ == "__main__":
#   fgru = FGRU(128, 64)

#   tensor = torch.rand(10, 5, 128)
#   x = fgru(tensor)
#   print(x.shape)