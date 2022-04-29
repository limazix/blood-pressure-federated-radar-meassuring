import torch


class Normalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, mean, std) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, data):
        return (data - self.mean)/self.std
