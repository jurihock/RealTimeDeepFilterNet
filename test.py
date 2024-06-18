import matplotlib.pyplot as plot
import numpy as np
import onnxruntime as ort
import torch

from audio import read, write
from cpx import CPX
from erb import ERB
from spectrum import spectrogram, erbgram
from stft import STFT

from df.enhance import init_df, df_features, enhance
from df.model import ModelParams
from df.utils import get_norm_alpha


def filter0(model, state, x):

    x = torch.from_numpy(x.astype(np.float32))
    y = enhance(model, state, x)
    y = y.detach().cpu().numpy()

    return y


def filter1(model, state, x):

    cpu = torch.device('cpu')

    if hasattr(model, 'reset_h0'):
        bs = x.shape[0]
        print(f'reset_h0 bs={bs}')
        model.reset_h0(batch_size=bs, device=cpu)
        assert False, 'TODO reset_h0'

    params = ModelParams()

    param_sr = params.sr # 48000
    param_fft_size = params.fft_size # 960
    param_hop_size = params.hop_size # 480
    param_fft_bins = params.fft_size // 2 + 1 # 481
    param_erb_bins = params.nb_erb # 32
    param_erb_min_width = params.min_nb_freqs # 2
    param_deep_filter_bins = params.nb_df # 96
    param_norm_alpha = get_norm_alpha(False) # 0.99

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
        erb_min_width=param_erb_min_width,
        deep_filter_bins=param_deep_filter_bins,
        norm_alpha=param_norm_alpha))
    print()

    stft = STFT(
        framesize=param_fft_size,
        hopsize=param_hop_size,
        window='hann')

    erb = ERB(
        samplerate=param_sr,
        fftsize=param_fft_size,
        erbsize=param_erb_bins,
        minwidth=param_erb_min_width,
        alpha=param_norm_alpha)

    cpx = CPX(
        cpxsize=param_deep_filter_bins,
        alpha=param_norm_alpha)

    spec, erb_feat, spec_feat = df_features(torch.from_numpy(x.astype(np.float32)), state, param_deep_filter_bins, device=cpu)
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
        exit()

    if False:

        x = state.erb_widths()
        y = erb.widths
        print(x)
        print(y)
        assert np.allclose(x, y)

        weights = erb.weights
        print(weights.shape)
        plot.figure()
        for i in range(weights.shape[-1]):
            plot.plot(weights[..., i])

        plot.show()
        exit()

    if False:

        x = torch.view_as_complex(spec).numpy()
        y = erb(x)

        foo = np.squeeze(erb_feat.numpy())
        bar = np.squeeze(y)

        erbgram(foo, name='foo', clim=(0,1))
        erbgram(bar, name='bar', clim=(0,1))

        plot.show()
        exit()

    if False:

        x = torch.view_as_complex(spec_feat).numpy()
        x = np.squeeze(np.abs(x))

        y = torch.view_as_complex(spec).numpy()
        y = cpx(y)
        y = np.squeeze(np.abs(y))

        erbgram(x, name='x', clim=(0,2))
        erbgram(y, name='y', clim=(0,2))

        plot.show()
        exit()

    if True:

        x = torch.view_as_complex(spec).numpy()
        y = erb(x)
        erb_feat = torch.from_numpy(y.astype(np.float32))

    if True:

        x = torch.view_as_complex(spec).numpy()
        y = cpx(x)
        y = np.stack((y.real, y.imag), axis=-1)
        spec_feat = torch.from_numpy(y.astype(np.float32))

    output = model(spec, erb_feat, spec_feat) # orig: spec.clone()
    enhanced = output[0].cpu()
    print('enhanced', enhanced.shape, enhanced.dtype)
    enhanced = enhanced.squeeze(1)
    print('enhanced squeeze', enhanced.shape, enhanced.dtype)
    enhanced = torch.view_as_complex(enhanced) # orig: as_complex
    print('enhanced complex', enhanced.shape, enhanced.dtype)
    print()

    y = state.synthesis(enhanced.detach().numpy())

    return y


if __name__ == '__main__':

    model, state, _ = init_df()

    model.eval()

    x, sr = read('x.wav', state.sr())

    y = filter1(model, state, x) \
        if True else \
        filter0(model, state, x)

    write('y.wav', sr, y)
