from typing import Union
from numpy.typing import ArrayLike, NDArray

import numpy as np
import warnings

from numpy.lib.stride_tricks import sliding_window_view


class STFT:
    """
    Short-Time Fourier Transform (STFT).
    """

    def __init__(self, framesize: int, *, hopsize: Union[int, None] = None, padsize: int = 0, center: bool = False, window: Union[bool, str, None] = True):
        """
        Create a new STFT plan.

        Parameters
        ----------
        framesize: int
            Time domain segment length in samples.
        hopsize: int, optional
            Distance between consecutive segments in samples.
            Defaults to `framesize // 4`.
        padsize: int, optional
            Number of zeros to pad the segments with.
        center: bool, optional
            Shift the zero-frequency to the center of the segment.
        window: bool, str, none, optional
            Window function name or a boolean, to enable the default hann window.
            Currently, only hann and rect window functions are supported.
        """

        assert framesize > 0
        assert hopsize > 0
        assert padsize >= 0

        if False: # TODO power of two warnings

            def is_power_of_two(n: int) -> bool:
                return (n != 0) and (n & (n-1) == 0)

            if not is_power_of_two(framesize):
                warnings.warn('The frame size should be a power of two for optimal performance!', UserWarning)

            if not is_power_of_two(framesize + padsize):
                warnings.warn('The sum of frame and pad sizes should be a power of two for optimal performance!', UserWarning)

        windows = {
            'rect':  lambda n: np.ones(n),
            'none':  lambda n: np.ones(n),
            'false': lambda n: np.ones(n),
            'true':  lambda n: np.hanning(n+1)[:-1],
            'hann':  lambda n: np.hanning(n+1)[:-1],
        }

        self.framesize = framesize
        self.hopsize   = hopsize or (self.framesize // 4)
        self.padsize   = padsize
        self.center    = center
        self.window    = windows[str(window).lower()](self.framesize)

    def freqs(self, samplerate: Union[int, None] = None) -> NDArray:
        """
        Returns an array of DFT bin center frequency values in hertz.
        If no sample rate is specified, then the frequency unit is cycles/second.
        """

        return np.fft.rfftfreq(self.framesize + self.padsize, 1 / (samplerate or 1))

    def stft(self, samples: ArrayLike) -> NDArray:
        """
        Estimates the DFT matrix for the given sample array.

        Parameters
        ----------
        samples: ndarray
            Array of time domain signal values.

        Returns
        -------
        dfts: ndarray
            Estimated DFT matrix of shape (samples, frequencies).
        """

        samples = np.atleast_1d(samples)

        assert samples.ndim == 1, f'Expected 1D array (samples,), got {samples.shape}!'

        frames = sliding_window_view(samples, self.framesize, writeable=False)[::self.hopsize]
        dfts   = self.fft(frames)

        return dfts

    def istft(self, dfts: ArrayLike) -> NDArray:
        """
        Synthesizes the sample array from the given DFT matrix.

        Parameters
        ----------
        dfts: ndarray
            DFT matrix of shape (samples, frequencies).

        Returns
        -------
        samples: ndarray
            Synthesized array of time domain signal values.
        """

        dfts = np.atleast_2d(dfts)

        assert dfts.ndim == 2, f'Expected 2D array (samples, frequencies), got {dfts.shape}!'

        gain = self.hopsize / np.sum(np.square(self.window))
        size = dfts.shape[0] * self.hopsize + self.framesize

        samples = np.zeros(size, float)

        frames0 = sliding_window_view(samples, self.framesize, writeable=True)[::self.hopsize]
        frames1 = self.ifft(dfts) * gain

        for i in range(min(len(frames0), len(frames1))):

            frames0[i] += frames1[i]

        return samples

    def fft(self, data: ArrayLike) -> NDArray:
        """
        Performs the forward FFT.
        """

        assert len(np.shape(data)) == 2

        data = np.atleast_2d(data) * self.window

        if self.padsize:

            data = np.pad(data, ((0, 0), (0, self.padsize)))

        if self.center:

            data = np.roll(data, self.framesize // -2, axis=-1)

        return np.fft.rfft(data, axis=-1, norm='forward')

    def ifft(self, data: ArrayLike) -> NDArray:
        """
        Performs the backward FFT.
        """

        assert len(np.shape(data)) == 2

        data = np.fft.irfft(data, axis=-1, norm='forward')

        if self.center:

            data = np.roll(data, self.framesize // +2, axis=-1)

        if self.padsize:

            data = data[..., :self.framesize]

        data *= self.window

        return data
