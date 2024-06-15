from typing import Tuple
from numpy.typing import ArrayLike

import matplotlib.pyplot as plot
import numpy as np


def spectrogram(x: ArrayLike, *,
                name: str = 'Spectrogram',
                xlim: Tuple[float, float] = (None, None),
                ylim: Tuple[float, float] = (None, None),
                clim: Tuple[float, float] = (-120, 0)):

    if not np.any(np.iscomplex(x)):

        x = np.atleast_1d(x)
        assert x.ndim == 1

        # X = STFT(framesize=..., hopsize=..., window=...).stft(x)
        # assert X.ndim == 2
        raise NotImplementedError('TODO STFT')

    else:

        X = np.atleast_2d(x)
        assert X.ndim == 2

    epsilon = np.finfo(X.dtype).eps

    spectrum = np.abs(X)
    spectrum = 20 * np.log10(spectrum + epsilon)

    timestamps  = np.arange(spectrum.shape[0]) # TODO np.arange(len(X)) * hopsize / samplerate
    frequencies = np.arange(spectrum.shape[1]) # TODO np.fft.rfftfreq(framesize, 1 / samplerate)

    extent = (timestamps[0], timestamps[-1], frequencies[0], frequencies[-1])
    args   = dict(aspect='auto', cmap='inferno', extent=extent, interpolation='nearest', origin='lower')

    plot.figure(name)
    plot.imshow(spectrum.T, **args)
    colorbar = plot.colorbar()

    plot.xlabel('time [s]')
    plot.ylabel('frequency [Hz]')
    colorbar.set_label('magnitude [dB]')

    plot.xlim(*xlim)
    plot.ylim(*ylim)
    plot.clim(*clim)

    return plot
