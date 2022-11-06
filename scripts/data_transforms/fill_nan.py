#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

class FillNan(object):

    def interpolate(self, data):
        data = pd.Series(data)
        data.interpolate(method='polynomial', order=4, inplece=True)
        return data.to_numpy()

    def __call__(self, data: np.array):
        for idx, line in enumerate(data):
            data[idx] = self.interpolate(line)
        return data
        