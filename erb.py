import numpy as np


def hz2erb(hz):
    """
    Converts frequency value in Hz to human-defined ERB band index,
    using the formula of Glasberg and Moore.
    """
    return 9.265 * np.log(1 + hz / (24.7 * 9.265))

def erb2hz(erb):
    """
    Converts human-defined ERB band index to frequency value in Hz,
    using the formula of Glasberg and Moore.
    """
    return 24.7 * 9.265 * (np.exp(erb / 9.265) - 1)


class ERB:

    def __init__(self, samplerate: int, fftsize: int, erbsize: int, minwidth: int):
        self.samplerate = samplerate
        self.fftsize = fftsize
        self.erbsize = erbsize
        self.minwidth = minwidth

        self.widths = ERB.get_band_widths(samplerate, fftsize, erbsize, minwidth)
        self.weights = ERB.get_band_weights(samplerate, self.widths)

    def __call__(self, dfts, db=True, alpha=0.99):
        x = np.abs(dfts)
        y = np.matmul(x, self.weights)
        if db:
            y = 20 * np.log10(y + np.finfo(dfts.dtype).eps)
            mean = np.full(y.shape[-1], y[..., 0, :])
            for i in range(y.shape[-2]):
                mean = y[..., i, :] * (1 - alpha) + mean * alpha
                y[..., i, :] -= mean
            y /= 40
        return y

    @staticmethod
    def get_band_widths(samplerate: int, fftsize: int, erbsize: int, minwidth: int):

        dftsize = fftsize / 2 + 1
        nyquist = samplerate / 2
        bandwidth = samplerate / fftsize

        erbmin = hz2erb(0)
        erbmax = hz2erb(nyquist)
        erbinc = (erbmax - erbmin) / erbsize

        bands = np.arange(1, erbsize + 1)
        freqs = erb2hz(erbmin + erbinc * bands)
        widths = np.round(freqs / bandwidth).astype(int)

        prev = 0
        over = 0

        for i in range(erbsize):

            next = widths[i]
            width = next - prev - over
            prev = next

            over = max(minwidth - width, 0)
            width = max(minwidth, width)

            widths[i] = width

        widths[erbsize - 1] += 1
        assert np.sum(widths) == dftsize

        return widths

    @staticmethod
    def get_band_weights(samplerate: int, widths: np.ndarray, normalized: bool = True, inverse: bool = False):

        n_freqs = int(np.sum(widths))
        all_freqs = np.linspace(0, samplerate // 2, n_freqs + 1)[:-1]

        b_pts = np.cumsum([0] + widths.tolist()).astype(int)[:-1]

        fb = np.zeros((all_freqs.shape[0], b_pts.shape[0]))

        for i, (b, w) in enumerate(zip(b_pts.tolist(), widths.tolist())):
            fb[b : b + w, i] = 1

        if inverse:
            fb = fb.t()
            if not normalized:
                fb /= np.sum(fb, axis=1, keepdim=True)
        else:
            if normalized:
                fb /= np.sum(fb, axis=0)

        return fb
