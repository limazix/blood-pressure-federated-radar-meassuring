#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from torch.utils.data import Dataset


class SubjectDataset(Dataset):
    """Class used to transform a subject data into a Pytorch Dataset"""

    def __init__(
        self, data: pd.DataFrame, transform=None, target_transform=None
    ) -> None:
        super().__init__()
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data.iloc[index, :2]
        y = self.data.iloc[index, 2]

        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)

        return X, y
