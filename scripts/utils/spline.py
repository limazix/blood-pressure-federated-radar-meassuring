#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import CubicSpline

class Spline:

    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.cubic_spline = None

    def build(self):
        X, Y = np.array([]), np.array([])
    
        for idx, value in enumerate(self.data):
            if not (np.isnan(value) or np.isinf(value)):
                np.append(X, [idx])
                np.append(Y, [value])
        
        if X.size > 10:
            self.cubic_spline = CubicSpline(X, Y)
            return True
        else:
            return False

    def __call__(self, x):
        if self.cubic_spline is not None:
            return self.cubic_spline(x)
