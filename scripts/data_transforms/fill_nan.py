import numpy as np


class FillNan(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, default_mean: np.float32) -> None:
        self.default_mean = default_mean

    def __call__(self, data):
        mean = np.nanmean(data, axis=0)
        mean = np.nan_to_num(mean, nan=self.default_mean)
        return np.nan_to_num(data, nan=mean)
