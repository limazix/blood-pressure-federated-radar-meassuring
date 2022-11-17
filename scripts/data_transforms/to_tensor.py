#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data: np.ndarray):
        return torch.from_numpy(data.astype(np.float32))
