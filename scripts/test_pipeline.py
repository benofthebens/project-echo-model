import torch


from src.modules.subband import MagnitudeMaskEstimator
from src.modules.denoiser import Denoiser
from src.modules.conv_block.fgru import FGRU

# run `python -m scripts.test_pipeline` from project root to execute

def test_full_pipeline():
    """
    Test the full audio processing pipeline by running a mock input through each stage sequentially
    """

    print(f'> beginning full pipeline test...')

    try:
        # alpha=0.3 is used for power law compression of audio features in Shetu et al.
        model = Denoiser(alpha=0.3)
        model.eval()
    except Exception as e:
        print(f'> error initializing Denoiser model: {e}')
        return
    
    # create mock audio (1 second of 16kHz audio, batch size of 2)
    x_waveform = torch.randn(2, 16000)

    print(f'> input waveform shape: {x_waveform}')
    print(f'> input waveform stats - min: {x_waveform.min().item():.4f}, max: {x_waveform.max().item():.4f}, mean: {x_waveform.mean().item():.4f}')

    with torch.no_grad():
        try:
            output = model(x_waveform)
            print(f'> full pipeline output shape: {output.shape}')

            if output.shape != x_waveform.shape:
                print(f'> output shape {output.shape} does not match input shape {x_waveform.shape}')
            else:
                print(f'> full pipeline test successful: output shape matches input shape')

            if not torch.isfinite(output).all():
                print(f'> warning: output contains non-finite values')
            else:
                print(f'> full pipeline test successful: output contains only finite values')

        except Exception as e:
            print(f'> error during full pipeline execution: {e}')
            raise e
        
        finally:
            print(f'> output waveform stats - min: {output.min().item():.4f}, max: {output.max().item():.4f}, mean: {output.mean().item():.4f}')
            print(f'> {output}')


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
    
    # create orchestrator model which calls all the substages


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
    # x = torch.randn(2, 64, 16000)
    # get_magnitude_mask(x)

    test_full_pipeline()