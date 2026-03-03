import torch.nn as nn

class LearnableSubbandSplitter(nn.Module):
    """
    Splits the input into frequency subbands using 1D convolutions.
    """
    def __init__(self, in_channels, num_subbands, kernel_size=3, padding=1):

        super().__init__()
        
        self.num_subbands = num_subbands
        self.in_channels = in_channels
        self.relu = nn.ReLU()

        # create list of 1d convolutions based on number of subbands
        self.conv_list = nn.ModuleList([
            nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=padding)
            for _ in range(num_subbands)
        ])

    def forward(self, x):
        """
        Forward pass for the adaptive subband splitting

        Args:
            x (torch.Tensor): input tensor;
                shape: (batch, in_channels, seq_len)

        Returns:
            list[torch.Tensor]: list of length `num_subbands` with elements of the following shape;
                shape: (batch, in_channels, seq_len)
        """
        # adaptive approach used as an alternative to manual subband chunking
        subbands = [self.relu(conv(x)) for conv in self.conv_list]
        return subbands