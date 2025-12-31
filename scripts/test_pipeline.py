import torch

from src.modules.subband_splitter import LearnableSubbandSplitter
from src.modules.subband_gru import SubbandedTemporalGRU
from src.modules.subband_fc_layers import FullyConnectedLayers

def get_magnitude_mask(x):
    """
    Stage 2 of the audio processing pipeline: subband spliting -> GRU -> FC layers 

    Args: 
        x (torch.Tensor): output from the pointwise conv of Stage 1;
            shape: (batch, channels, seq_len)

    Returns:
        torch.Tensor: estimated magnitude mask;
            shape: (batch, out_channels)
    """
    IN_CHANNELS = 64  # output of the pointwise convolution
    NUM_SUBBANDS = 2
    HIDDEN_SIZE = 128

    # step 1 | split input into subbands dynamically
    splitter = LearnableSubbandSplitter(in_channels=IN_CHANNELS, num_subbands=NUM_SUBBANDS)
    subbands = splitter(x)
    print(f'splitter output shape: {len(subbands)} subbands of shape {subbands[0].shape}')
    
    # step 2 | create one GRU per subband
    grus = torch.nn.ModuleList([
        SubbandedTemporalGRU(input_size=IN_CHANNELS, hidden_size=HIDDEN_SIZE)
        for subband in range(NUM_SUBBANDS)
    ])

    gru_output = []

    for i, subband_tensor in enumerate(subbands):
        # transpose from shape (batch, channels, seq_len) to (batch, seq_len, channels)
        gru_input = subband_tensor.transpose(1, 2)

        # get output for each subbanded gru
        output = grus[i](gru_input)
        print(f'subbband #{i} gru shape: {output.shape}')
        gru_output.append(output)

    # step 3 | concatenate features
    concatenated = torch.cat(gru_output, dim=-1)
    print(f'concatenated features shape: {concatenated.shape}')

    # step 4 | estimate magnitude mask with FC layers
    total_features = concatenated.shape[-1]
    fc_layers = FullyConnectedLayers(in_channels=total_features)
    magnitude_mask = fc_layers(concatenated)

    print(f'final mask shape: {magnitude_mask.shape}')
    return magnitude_mask

if __name__ == '__main__':
    
    # mock 1 second of 16kHz audio, with batch size 2 & 64 channels
    x = torch.randn(2, 64, 16000)
    get_magnitude_mask(x)