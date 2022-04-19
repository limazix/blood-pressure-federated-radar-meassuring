#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class ArctanDemodulation(object):
    def __call__(self, data):
        data = np.arctan2(data.T[1], data.T[0])
        return data
