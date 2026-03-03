import torch
import torch.nn as nn

class SubbandedTemporalGRU(nn.Module):
    """
    Perform sequential modelling on an input sequence
    """
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True, bidirectional=True):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional
        )

    def forward(self, x):
        """
        Forward pass for temporal GRU

        Args:
            x (torch.Tensor): input tensor;
              shape: (batch, seq_len, input_size)

        Returns:
            torch.Tensor: features;
                shape: (batch, seq_len, hidden_size * directions)
        """
        # h_n is the initial hidden state with shape (num_layers * directions, batch_size, hidden_size)
        output, h_n = self.gru(x)
        return output