#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

from torch import nn


class FLAgent(ABC):
    """Class used to abstract the federated learning agent methods"""

    def __init__(self, model: nn.Module, learning_rate=0.0001) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    @abstractmethod
    def fit(self, X, y):
        """Method used to train the model

        Parameters:
            X (list): Multidimensional input matrix
            y (list): Multidimensional expected output matrix
        """

    def get_params(self):
        """Method used to get all model weights

        Returns:
            list: Multidimensional weights matrix
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
