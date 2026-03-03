import torch
from src.modules.subband import MagnitudeMaskEstimator

# run `python -m scripts.test_pipeline` from project root to execute

def get_magnitude_mask(x):
    """
    Stage 2 of the audio processing pipeline: subband spliting -> GRU -> FC layers -> mask estimation

    Args: 
        x (torch.Tensor): output from the pointwise conv of Stage 1;
            shape: (batch, channels, seq_len)

    Returns:
        torch.Tensor: estimated magnitude mask;
            shape: (batch, out_channels, seq_len)
    """
    print(f'> beginning magnitude mask estimation...')
    model = MagnitudeMaskEstimator()

    # set model to evaluation mode so the batch norm uses pre-learned data each time 
    model.eval()
    # deactivate gradient calculations to save memory
    with torch.no_grad():
        magnitude_mask = model(x)
  
    # usable audio values should be non-negative; sigmoid bounds the output to [0,1]
    print(f'> mask min value: {magnitude_mask.min().item():.4f}')
    print(f'> mask max value: {magnitude_mask.max().item():.4f}')
    # untrained model should produce values ~0.5
    print(f'> mask mean: {magnitude_mask.mean().item():.4f}')
    print(f'> final mask shape: {magnitude_mask.shape}')

if __name__ == '__main__':
    # use batch size of 2 and 64 channels to mock 1 second of 16kHz audio
    x = torch.randn(2, 64, 16000)
    get_magnitude_mask(x)

    # add FGRU test here 