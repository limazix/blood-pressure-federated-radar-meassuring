import numpy as np
from scipy.stats import hmean


class FillNan(object):

    def __init__(self, default_mean: np.float32) -> None:
        self.default_mean = default_mean

    def __call__(self, data):
        mean = hmean(data, axis=0, nan_policy='omit')
        mean = np.nan_to_num(mean, nan=self.default_mean)
        return np.nan_to_num(data, nan=mean)
