import torch
import torch.nn as nn

class FullyConnectedLayers(nn.Module):
    """
    Creates FC layers, with output used to help estimate magnitude mask
    """
    def __init__(
            self,
            in_channels,
            neurons=[257,257],  # specified in (Shetu et al.)
            out_channels=5):    # hyperparameter, will need finetuning
        
        super().__init__()

        self.fc1 = nn.Linear(in_channels, neurons[0])
        self.fc2 = nn.Linear(neurons[0], neurons[1])
        self.output = nn.Linear(neurons[1], out_channels)

        self.batch_norm1 = nn.BatchNorm1d(neurons[0])
        self.batch_norm2 = nn.BatchNorm1d(neurons[1])

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass for the fully connected (mask estimation) layers
      
        Args:
          x (torch.Tensor): Concatenated features from GRU blocks;
            shape: (batch_size, seq_length, in_channels)

        Returns:
          torch.Tensor: Estimated magnitude mask;
            shape: (batch_size, out_channels)
        """
        # use global average pooling to flatten the tensor irrespective of sequence length
        # new shape preserves batch dimension: (batch_size, channels)
        x = torch.mean(x, dim=1)

        # apply FC layers with batch norm and ReLU across features
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.relu(self.batch_norm2(self.fc2(x)))

        return self.output(x)