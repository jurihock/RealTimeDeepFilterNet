import numpy as np


class CPX:

    def __init__(self, cpxsize: int, alpha: float):

        self.cpxsize = cpxsize
        self.alpha = alpha

    def __call__(self, dfts):

        y = np.copy(dfts[..., :self.cpxsize])

        # TODO ISSUE #100
        mean = np.full(y.shape[-1], y[..., 0, :])
        alpha = self.alpha
        for i in range(y.shape[-2]):
            mean = np.absolute(y[..., i, :]) * (1 - alpha) + mean * alpha # orig: norm
            y[..., i, :] /= np.sqrt(mean)

        return y
