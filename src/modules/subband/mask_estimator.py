import torch
import torch.nn as nn

# import from the same directory
from .splitter import LearnableSubbandSplitter
from .gru import SubbandedTemporalGRU
from .fc_layers import FullyConnectedLayers

FC_SIZE = [257,257]

class MagnitudeMaskEstimator(nn.Module):
    """
    Implement three-part mask estimation pipeline using subband splitting, temporal GRUs and FC layers
    """
    def __init__(
            self,
            in_channels=64,          # assumed output of pointwise conv layer
            num_subbands=2,          # number of subband temporal GRU blocks, specified in (Shetu et al.)
            hidden_size=128,         # size of GRU layers, specified in (Shetu et al.)
            num_layers=2,            # two GRU layers, specified in (Shetu et al.)
            bidirectional=False,     # for live processing, GRU should be unidirectional
            neurons=FC_SIZE,         # size of FC hidden layers, specified in (Shetu et al.)
            out_channels=FC_SIZE[0]  # output the intermediate real magnitude mask, specified in (Shetu et al.)
        ):

        super().__init__()

        self.num_directions = 2 if bidirectional else 1
        # determine total number of features from GRU outputs
        total_features = num_subbands * hidden_size * self.num_directions

        self.splitter = LearnableSubbandSplitter(in_channels=in_channels, num_subbands=num_subbands)

        self.grus = nn.ModuleList([
            SubbandedTemporalGRU(input_size=in_channels,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 batch_first=True,
                                 bidirectional=bidirectional
                                 )
            for _ in range(num_subbands)
        ])

        self.fc_layers = FullyConnectedLayers(in_channels=total_features, neurons=neurons, out_channels=out_channels)

    def forward(self, x):
        """
        Forward pass of the orchestrator of the subband stage

        Args: x (torch.Tensor): input tensor;
            shape: (batch, channels, seq_len)

        Returns:
            torch.Tensor: magnitude mask estimation
                shape: (batch, out_channels, seq_len)
        """
        # step 1 | split input into subbands dynamically
        subbands = self.splitter(x)

        # step 2 | process each subband with its own GRU
        gru_output = []

        for i, subband_tensor in enumerate(subbands):
            # transpose from shape (batch, channels, seq_len) to (batch, seq_len, channels)
            gru_input = subband_tensor.transpose(1, 2)
            # append output from each subbanded gru
            gru_output.append(self.grus[i](gru_input))

        # step 3 | concatenate features and estimate mask
        concatenated = torch.cat(gru_output, dim=-1)
        fc_output = self.fc_layers(concatenated)

        # force output mask to be in range [0,1] 
        return torch.sigmoid(fc_output)
