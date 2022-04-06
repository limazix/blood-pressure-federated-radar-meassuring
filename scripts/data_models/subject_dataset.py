#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset


class SubjectDataset(Dataset):
    """Class used to transform a subject data into a Pytorch Dataset

    Parameters:
        radar (Numpy.array):
        radar_sr (int): Radar data sample rate
        bp (Numpy.array):
        bp_sr (int): Blood pressure sample rate
        window_size (float): Window size in seconds
        overlap (float): Window overlap in seconds
    """

    def __init__(
        self,
        radar,
        radar_sr,
        bp,
        bp_sr,
        window_size,
        overlap,
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.overlap = overlap
        self.transform = transform
        self.target_transform = target_transform
        self.input = self.setup(radar, radar_sr)
        self.output = self.setup(bp, bp_sr)

    def setup(self, data, sr):
        """Method used to creates the windows

        Parameters:
            data (Numpy.array): Data to be processed
            sr (int): Data sample rate

        Returns:
            Numpy.array: List of windows
        """
        windows = []
        start = 0
        end = sr * self.window_size
        while end < len(data):
            window = data[start:end]
            windows.append(window)
            start = start + int((self.window_size - self.overlap) * sr)
            end = start + int(sr * self.window_size)
        return windows

    def prune(self, start_index, end_index):
        self.input = self.input[start_index:end_index]
        self.output = self.output[start_index:end_index]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        X = self.input[index]
        y = self.output[index]

        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)

        return torch.Tensor(X), torch.Tensor(y)
