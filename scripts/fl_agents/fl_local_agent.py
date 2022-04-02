#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
from torch.utils.data import DataLoader

from .fl_agent import FLAgent


class FLLocalAgent(FLAgent):
    """Class used to represent the architecture local (in-device) agent"""

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        learning_rate=0.0001,
    ) -> None:
        super().__init__(model, learning_rate)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def fit(self, X, y):
        pass
