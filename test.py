import matplotlib.pyplot as plot
import numpy as np
import onnxruntime as ort
import torch

from audio import read, write
from spectrum import spectrogram
from stft import STFT

from df.enhance import enhance, init_df, df_features
from df.model import ModelParams


def filter(model, state, x):

    device = torch.device('cpu')

    model.eval()

    if hasattr(model, 'reset_h0'):
        bs = x.shape[0]
        print(f'reset_h0 bs={bs}')
        model.reset_h0(batch_size=bs, device=device)
        assert False, 'TODO reset_h0'

    params = ModelParams()

    param_sr = params.sr
    param_fft_size = params.fft_size
    param_hop_size = params.hop_size
    param_fft_bins = params.fft_size // 2 + 1
    param_erb_bins = params.nb_erb # 32
    param_deep_filter_bins = params.nb_df # 96

    assert getattr(model, 'freq_bins', param_fft_bins) == param_fft_bins
    assert getattr(model, 'erb_bins', param_erb_bins) == param_erb_bins
    assert getattr(model, 'nb_df', getattr(model, 'df_bins', param_deep_filter_bins)) == param_deep_filter_bins
    assert state.sr() == param_sr
    assert len(state.erb_widths()) == param_erb_bins

    print(dict(
        sr=param_sr,
        fft_size=param_fft_size,
        hop_size=param_hop_size,
        fft_bins=param_fft_bins,
        erb_bins=param_erb_bins,
        deep_filter_bins=param_deep_filter_bins))
    print()

    stft = STFT(framesize=param_fft_size, hopsize=param_hop_size, window='hann')

    spec, erb_feat, spec_feat = df_features(x, state, param_deep_filter_bins, device=device)
    print('spec', spec.shape, spec.dtype)
    print('erb_feat', erb_feat.shape, erb_feat.dtype)
    print('spec_feat', spec_feat.shape, spec_feat.dtype)
    print()

    if False:
        dfts0 = np.squeeze(torch.view_as_complex(spec).numpy())
        print(dfts0.shape, dfts0.dtype)
        spectrogram(dfts0, name='dfts0')
        dfts1 = stft.stft(x[0])
        print(dfts1.shape, dfts1.dtype)
        spectrogram(dfts1, name='dfts1')
        plot.show()

    enhanced = model(spec, erb_feat, spec_feat)[0].cpu() # orig: spec.clone()
    print('enhanced', enhanced.shape, enhanced.dtype)
    enhanced = enhanced.squeeze(1)
    print('enhanced squeeze', enhanced.shape, enhanced.dtype)
    enhanced = torch.view_as_complex(enhanced) # orig: as_complex
    print('enhanced complex', enhanced.shape, enhanced.dtype)
    print()

    y = torch.as_tensor(state.synthesis(enhanced.detach().numpy()))

    return y


if __name__ == '__main__':

    model, state, _ = init_df()

    x, sr = read('x.wav', state.sr())

    x = torch.from_numpy(x)
    y = filter(model, state, x)
    y = y.detach().cpu().numpy()

    write('y.wav', sr, y)
