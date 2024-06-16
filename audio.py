import numpy as np
import resampy
import soundfile

from numpy.typing import NDArray


def read(path: str, sr: int) -> NDArray:

    samples, samplerate = soundfile.read(path, always_2d=True)

    if samplerate != sr:
        samples = resampy.resample(samples, samplerate, sr)
        samplerate = sr

    samples = samples.T

    return samples, samplerate


def write(path: str, sr: int, samples: NDArray):

    samples = np.squeeze(samples.T)
    soundfile.write(path, samples, sr)
