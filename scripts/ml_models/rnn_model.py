#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn


class RNNModel(nn.Module):
    """Class used to represent a simple RNN implementation in Pytorch for Radar

    Parameters:
        input_size (int): The size of the input data array
        hidden_size (int): Number of hidden layers
        output_size (int): The size of the output data array
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
    ) -> None:
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.setup_layers(input_size, hidden_size, num_layers, output_size)

    def setup_layers(self, input_size, hidden_size, num_layers, output_size):
        rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        fc = nn.Linear(hidden_size, output_size)
        self.layers = nn.ModuleList([rnn, fc])

    def forward(self, X):
        """Method used to convert the in-phase (I) and quadrature (Q) radar signals to the correspondent blood pressure

        Parameters:
            X (array): Bi-dimensional array with I and Q values
        Returns:
            (array): Uni-dimensional array of blood pressures
        """
        out, _ = self.layers[0](X.float())

        out = self.layers[1](out)
        return out
