import torch
import torch.nn as nn

class FullyConnectedLayers(nn.Module):
    """
    Creates FC layers, with output used to help estimate magnitude mask
    """
    def __init__(self, in_channels, neurons, out_channels):
        
        super().__init__()

        # use 1d conv to preserve order of dimensions between batch norming
        self.fc1 = nn.Conv1d(in_channels, neurons[0], kernel_size=1)
        self.fc2 = nn.Conv1d(neurons[0], neurons[1], kernel_size=1)
        self.output = nn.Conv1d(neurons[1], out_channels, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(neurons[0])
        self.batch_norm2 = nn.BatchNorm1d(neurons[1])

        self.relu = nn.ReLU()

    def swap_time_and_feature_dims(self, x):
        """
        Helper method to swap the time (seq_len) and channels dimensions between Linear layers and BatchNorm layers
        """
        return x.transpose(1, 2)

    def forward(self, x):
        """
        Forward pass for the fully connected (mask estimation) layers
      
        Args:
          x (torch.Tensor): concatenated features from GRU blocks;
            shape: (batch_size, seq_len, in_channels)

        Returns:
          torch.Tensor: estimated magnitude mask;
            shape: (batch_size, out_channels, seq_len)
        """

        # convert from (batch, time, features) to (batch, features, time)
        x = self.swap_time_and_feature_dims(x)

        # apply FC layers with batch norm and ReLU across features
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        
        return self.output(x) 