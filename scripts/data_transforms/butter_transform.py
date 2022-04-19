#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy import signal

from utils.configurator import config


class ButterTransform(object):
    def __init__(self) -> None:
        self.config = config["butterworth"]

    def __call__(self, data):
        nyq = 0.5 * int(config["dataset"]["radar_sr"])
        low = float(self.config["low_freq"]) / nyq
        high = float(self.config["high_freq"]) / nyq
        sos = signal.butter(
            N=int(self.config["order"]),
            Wn=[low, high],
            btype=self.config["type"],
            output=self.config["output"]
        )
        data[0] = signal.sosfilt(sos, data[0])
        data[1] = signal.sosfilt(sos, data[1])
        return data
