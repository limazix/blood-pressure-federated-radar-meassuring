#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from utils.spline import Spline


class FillNan(object):
    def __init__(self, splines: np.ndarray) -> None:
        self.splines = splines

    def get_value(self, x, y, z):
        if np.isnan(y) or np.isinf(y):
            return self.splines[z](x)
        else:
            return y

    def transform_unidimensional(self, data, z=0):
        spline = Spline(data)
        if spline.build():
            self.splines[z] = spline
        return np.array([self.get_value(x, y, z) for x, y in enumerate(data)])

    def __call__(self, data: np.ndarray):
        if self.splines.size > 1:
            for z, d in enumerate(data):
                data[z] = self.transform_unidimensional(d, z)
        else:
            data = self.transform_unidimensional(data)

        return data
